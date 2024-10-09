import copy
import time
import uuid
from typing import Dict, List, Optional
from tinyllm.common.shared_arr import SharedInt
from tinyllm.common.io_struct import Batch
from tinyllm.common.token_load import TokenLoad
from tinyllm.common.io_struct import Req
from tinyllm.utils.log_utils import logger
from tinyllm.common.backend.continues_backend import ContinuesBatchBackend
from tinyllm.common.queue.continues_queue import ContinuesBatchQueue
from tinyllm.common.io_struct import NormalReq, FinishStatus
class OfflineManager:
    def __init__(self, args, prompt_ids: List[List[int]], sampling_params):
        self.args = args
        self.model_weightdir = args.model_dir
        self.world_size = args.tp
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        # 用共享内存进行共享，router 模块读取进行精确的调度估计
        self.shared_can_use_token_num = SharedInt(f"{args.nccl_port}_mem_manger_can_use_token_num")

        self.shared_token_load = TokenLoad(f"{str(args.nccl_port)}_shared_token_load")
        self.shared_token_load.set_current_load(0.0)
        self.shared_token_load.set_logical_max_load(0.0)
        self.shared_token_load.set_dynamic_max_load(0.0)

        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        
        self.prompt_ids = prompt_ids
        self.prompts_num = len(prompt_ids)
        self.sampling_params = sampling_params

    def start_model(self):
        self.model_lpcs = []
        for i in range(self.world_size):
            self.model_lpcs.append([ModelLpc()])
        for rank_id in range(self.world_size):
            kvargs = {
                "args": self.args,
                "rank_id": rank_id,
                "world_size": self.world_size,
                "weight_dir": self.model_weightdir,
                "load_way": self.load_way,
                "max_total_token_num": self.max_total_token_num,
                "mode": self.mode,
                "max_req_num": self.args.running_max_req_size + 8,
                "max_seq_length": self.args.max_req_total_len + 8,  # 留一点余量
                "nccl_port": self.args.nccl_port,
                "data_type": self.args.data_type,
                "eos_id": self.eos_id,
            }
            self.model_lpcs[rank_id].init_model(kvargs)
        
        self.req_queue = ContinuesBatchQueue(self.args, self)

    def init_reqs(self):
        for group_id, prompt_id in enumerate(self.prompt_ids):
            req = NormalReq(group_id, prompt_id, self.sampling_params)
            self.req_queue.append(req)

        return

    def loop_for_fwd(self):
        while True: # zeyu 需要用是否有prompt还没有处理完进行判断
            is_empty = self.step()
            if is_empty:
                break

    def step(self):
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is None:
                return True
            else:
                self.running_batch = new_batch
                self._prefill_batch(self.running_batch)
                self._filter_running_batch(self.running_batch)
                self.has_wait_tokens = 0
            return False
        
        if self.has_wait_tokens >= self.max_total_token_num:
            new_mini_batch = self.req_queue.generate_newbatch(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                # TODO zeyu ignore stats now
                self._merge_batch(self.running_batch, new_mini_batch)
                self.running_batch.merge(new_mini_batch)
                return False

        if self._can_decode(self.running_batch):
            self._decode_batch(self.running_batch)
            self._filter_running_batch()
            self.has_wait_tokens += 1
            return False
        else:
            paused_reqs = select_paused_reqs(
                self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num
            )
            self._pause_reqs(self.running_batch, paused_reqs)
            self.has_wait_tokens = 0
            return False

    def _init_batch(self, batch):
        ans = [self.model_lpcs[i].init_batch(batch.batch_id, batch.reqs) for i in range(self.world_size)]
        self._update_init_status_to_batch(batch, ans[0])

    def _prefill_batch(self, batch):
        self._init_batch(batch)
        ans = [self.model_lpcs[i].prefill_batch(batch.batch_id) for i in range(self.world_size)] 

        self._update_out_status_to_batch(batch, ans[0])
        unifinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
        batch.filter_out_finished_reqs(unifinished_req_ids, finished_req_ids)
        self._handle_finished_reqs(batch, unifinished_req_ids, finished_req_ids)
    
    def _decode_batch(self, batch):
        ans = [self.model_lpcs[i].decode_batch(batch.batch_id) for i in range(self.world_size)]
        self._update_out_status_to_batch(batch, ans[0])
        unifinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
        batch.filter_out_finished_reqs(unifinished_req_ids, finished_req_ids)
        self._handle_finished_reqs(batch, unifinished_req_ids, finished_req_ids)
    
    def _filter_batch(self, batch, unifinished_req_ids, finished_req_ids):
        for i in range(self.world_size):
            self.model_lpcs[i].filter_batch(batch.batch_id, unifinished_req_ids, finished_req_ids)
        return
    
    def _merge_batch(self, batch1, batch2):
        for i in range(self.world_size):
            self.model_lpcs[i].merge_batch(batch1.batch_id, batch2.batch_id)
        return
    
    def _remove_batch(self, batch):
        for i in range(self.world_size):
            self.model_lpcs[i].remove_batch(batch.batch_id)
        return
    
    def _pause_reqs(self, batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.seq_len) for r in pasue_reqs]
        for i in range(self.world_size):
            self.model_lpcs[i].pause_reqs(batch.batch_id, pasue_reqs_info)
        return
    
    def _handle_finished_reqs(self, batch, unifinished_req_ids, finished_req_ids):
        if len(finished_req_ids) != 0:
            if batch.is_clear():
                self._remove_batch(batch)
            else:
                self._filter_batch(batch, unifinished_req_ids, finished_req_ids)
        return
    
    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _update_init_status_to_batch(self, batch, status):
        self._update_out_status_to_batch(batch, status)
        return
    
    def _update_out_status_to_batch(self, batch, status):
        for req_id, (req_status, cur_kv_len, cur_output_len, token_info_list, finish_status_value, extral_info) in status.items():
            req: Req = batch.id_to_reqs[req_id]
            req.req_status = req_status
            req.cur_kv_len = cur_kv_len
            req.cur_output_len = cur_output_len
            if not req.finish_status.is_aborted():
                req.finish_status = FinishStatus(finish_status_value)
            new_batch_decode_need_tokens += req.get_decode_need_tokens()

        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return
    
    def _can_decode(self, batch):
        return batch.batch_decode_need_tokens + self.get_used_tokens() <= self.max_total_token_num
    
    def get_used_tokens(self):
        return self.max_total_token_num - self.shared_can_use_token_num.get_value()
    



class ModelLpc():
    def init_model(self, kvargs):
        self.world_size = kvargs["world_size"]
        self.backend = ContinuesBatchBackend()
        logger.info(f"use {self.backend.__class__.__name__}")
        self.backend.init_model(kvargs)

        return

    def add_batch(self, batch_id, reqs):
        return self.backend.add_batch(batch_id, reqs)

    def prefill_batch(self, batch_id):
        return self.backend.prefill_batch(batch_id)

    def decode_batch(self, batch_id):
        return self.backend.decode_batch(batch_id)

    def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        return self.backend.filter_batch(batch_id, req_id_list, finished_req_id_list)

    def pause_reqs(self, batch_id, req_list):
        return self.backend.pause_reqs(batch_id, req_list)

    def merge_batch(self, batch_id1, batch_id2):
        return self.backend.merge_batch(batch_id1, batch_id2)

    def remove_batch(self, batch_id):
        return self.backend.remove_batch(batch_id)
            