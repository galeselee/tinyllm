import torch
import numpy as np
from typing import List
from tinyllm.common.basemodel.triton_kernel.apply_penalty import apply_penalty
from tinyllm.common.backend.base_backend import ModeBackend
from tinyllm.utils.infer_utils import calculate_time
from tinyllm.common.infer_batch import InferBatch, InferReq, requests_mapping
from tinyllm.common.io_struct import ReqRunStatus

def _sample(logits, reqs, eos_id: List[int] = [2]):
    logits = logits.contiguous()
    (
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        exponential_decay_length_penalties,
        temperatures,
        top_ps,
        top_ks,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
        length_penalty_idx,
        mask_eos_reqs,
    ) = _get_post_sample_tensors(reqs)

    apply_penalty(
        logits,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
    )
    logits[:, eos_id] = logits[:, eos_id] + torch.abs(logits[:, eos_id]) * (
        torch.pow(exponential_decay_length_penalties, length_penalty_idx).view((-1, 1)) - 1
    )
    if mask_eos_reqs.any():
        logits[mask_eos_reqs, eos_id] = -1000000.0
    logits.div_(temperatures.view((-1, 1)))
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = _top_p_top_k(probs, top_ps, top_ks)
    sampled_index = torch.multinomial(probs_sort, num_samples=1, replacement=True)

    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)
    batch_next_token_probs = torch.gather(probs_sort, dim=1, index=sampled_index)

    return batch_next_token_ids.view(-1), batch_next_token_probs.view(-1)


def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    probs_sort[torch.arange(0, probs.shape[-1], device="cuda").view(1, -1) >= top_ks.view(-1, 1)] = 0.0

    return probs_sort, probs_idx


def _get_post_sample_tensors(reqs):
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    repetition_penalties: List[float] = []
    exponential_decay_length_penalties: List[float] = []
    temperatures: List[float] = []
    top_ps: List[float] = []
    top_ks: List[int] = []
    p_token_ids: List[int] = []
    p_token_counts: List[int] = []
    p_seq_len: List[int] = [
        0,
    ]
    p_max_len_in_batch: int = 0
    length_penalty_idx: List[int] = []
    mask_eos_reqs: List[bool] = []
    for i, req_obj in enumerate(reqs):
        id_to_count = req_obj.out_token_id_count
        sample_param = req_obj.sampling_param
        presence_penalties.append(sample_param.presence_penalty)
        frequency_penalties.append(sample_param.frequency_penalty)
        repetition_penalties.append(sample_param.repetition_penalty)
        exponential_decay_length_penalties.append(sample_param.exponential_decay_length_penalty[1])
        out_token_len = len(req_obj.input_token_ids) - req_obj.prompt_len
        length_penalty_idx.append(max(out_token_len - sample_param.exponential_decay_length_penalty[0], 0))
        mask_eos_reqs.append(out_token_len < sample_param.min_new_tokens - 1)

        temperatures.append(sample_param.temperature)
        top_ps.append(sample_param.top_p)
        top_ks.append(sample_param.top_k)

        for token_id, count in id_to_count.items():
            p_token_ids.append(token_id)
            p_token_counts.append(count)
        p_seq_len.append(len(id_to_count))
        p_max_len_in_batch = max(p_max_len_in_batch, len(id_to_count))

    presence_penalties = torch.tensor(presence_penalties, dtype=torch.float, device="cuda")
    frequency_penalties = torch.tensor(frequency_penalties, dtype=torch.float, device="cuda")
    repetition_penalties = torch.tensor(repetition_penalties, dtype=torch.float, device="cuda")
    exponential_decay_length_penalties = torch.tensor(
        exponential_decay_length_penalties, dtype=torch.float, device="cuda"
    )
    temperatures = torch.tensor(temperatures, dtype=torch.float, device="cuda")
    top_ps = torch.tensor(top_ps, dtype=torch.float, device="cuda")
    top_ks = torch.tensor(top_ks, dtype=torch.int32, device="cuda")
    p_token_ids = torch.tensor(p_token_ids, dtype=torch.int32, device="cuda")
    p_token_counts = torch.tensor(p_token_counts, dtype=torch.int32, device="cuda")
    p_seq_len = torch.tensor(p_seq_len, dtype=torch.int32, device="cuda")
    p_cumsum_seq_len = torch.cumsum(p_seq_len, dim=0, dtype=torch.int32)
    length_penalty_idx = torch.tensor(length_penalty_idx, dtype=torch.int32, device="cuda")
    mask_eos_reqs = torch.tensor(mask_eos_reqs, dtype=torch.bool, device="cuda")
    return (
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        exponential_decay_length_penalties,
        temperatures,
        top_ps,
        top_ks,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
        length_penalty_idx,
        mask_eos_reqs,
    )

def prepare_prefill_inputs(batch: InferBatch):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    for request_id in batch.request_ids:
        req: InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING

        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)

        seq_len = len(req.input_token_ids)
        input_token_len = seq_len - req.cur_kv_len

        input_id = req.input_token_ids[req.cur_kv_len :]

        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)
        b_ready_cache_len.append(req.cur_kv_len)
        start_loc += input_token_len

    input_ids = np.concatenate(input_ids, dtype=np.int64)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")
    kwargs = {
        "batch_size": len(batch),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "b_ready_cache_len": b_ready_cache_len,
        "is_prefill": True,
    }

    return kwargs, run_reqs


# @calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(batch: InferBatch):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for request_id in batch.request_ids:
        req: InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        input_id = req.input_token_ids[-1]
        seq_len = len(req.input_token_ids)
        assert req.cur_kv_len == seq_len - 1
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    kwargs = {
        "batch_size": len(batch),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "is_prefill": False,
    }
   
    return kwargs, run_reqs


class ContinuesBatchBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def forward(self, batch_id, is_prefill):
        # special code for return all prompt_logprobs
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.is_multimodal)
        else:
            kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)

        logits = self.model.forward(**kwargs)
        next_token_ids, next_token_probs = _sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            metadata = {
                "id": int(next_token_id),
                "logprob": float(next_token_logprob),
            }
            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [(int(next_token_id), metadata)],
                req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                None,
            )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数

        self.cache[batch.batch_id] = batch
        return output_dict
