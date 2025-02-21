import os
import json
import torch
from typing import final

from tinyllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from tinyllm.common.basemodel.infer_struct import InferStateInfo
from tinyllm.common.managers.mem_manager import MemoryManager
from tinyllm.common.managers.req_manager import ReqManager
from tinyllm.utils.infer_utils import init_req_to_token_indexes
from tinyllm.utils.build_utils import repair_config
from tinyllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req

torch.backends.cudnn.enabled = True


class TpPartBaseModel:
    # weight class
    pre_and_post_weight_class = None
    transformer_weight_class = None

    # infer class
    pre_layer_infer_class = None
    post_layer_infer_class = None
    transformer_layer_infer_class = None

    # infer state class
    infer_state_class = InferStateInfo

    def __init__(self, kvargs):
        self.tp_rank_ = kvargs["tp_rank"]
        self.world_size_ = kvargs["world_size"]
        self.weight_dir_ = kvargs["weight_dir"]
        self.max_total_token_num = kvargs["max_total_token_num"]
        self.load_way = kvargs.get("load_way", "HF")
        self.weight_dict = kvargs.get("weight_dict", None)
        self.max_req_num = kvargs.get("max_req_num", 1000)
        self.max_seq_length = kvargs.get("max_seq_length", 1024 * 5)
        # is_token_healing 和 return_all_prompt_logics 是有排斥关系的两个模式，只能单独有一个生效
        # 主要是在prefill阶段返回多少个token的用于后续处理相关。
        self.is_token_healing = kvargs.get("is_token_healing", False)
        self.return_all_prompt_logics = kvargs.get("return_all_prompt_logics", False)
        assert not (self.is_token_healing and self.return_all_prompt_logics), "can not be true in same time"
        self.data_type = kvargs.get("data_type", "float16")

        self._init_datatype()
        self._init_config()
        self._verify_attn_heads_tensor_paralell()
        self._verify_load_weight()
        self._init_weights()
        self._init_mem_manager()
        self._init_req_manager()
        self._init_infer_layer()
        self._init_network_hyperparams()
        self._init_custom()
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])

        return

    def _verify_attn_heads_tensor_paralell(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    def _verify_load_weight(self):
        assert self.load_way == "HF", "only support HF format weights"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.tp_rank_, self.world_size_, self.data_type, network_config=self.config, mode=self.mode
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i, self.tp_rank_, self.world_size_, self.data_type, network_config=self.config, mode=self.mode
            )
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        self.mem_manager = MemoryManager(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_attention_heads"] // self.world_size_,
            head_dim=self.config["n_embed"] // self.config["num_attention_heads"],
            layer_num=self.config["n_layer"],
        )
        return

    def _init_req_manager(self):
        self.req_manager = ReqManager(self.max_req_num, self.max_seq_length, self.mem_manager)
        return

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.post_infer = self.post_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i, tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
            )
            for i in range(self.config["n_layer"])
        ]
        return

    def _init_network_hyperparams(self):
        # Dealing with head_dim_!=n_embed // num_attention_heads scenarios, such as mistral 13B
        head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.head_dim_ = self.config.get("head_dim", head_dim_)
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return

    def _init_datatype(self):
        if self.data_type in ["fp16", "float16"]:
            self.data_type = torch.float16
        elif self.data_type in ["bf16", "bfloat16"]:
            self.data_type = torch.bfloat16
        elif self.data_type in ["fp32", "float32"]:
            self.data_type = torch.float32
        else:
            raise ValueError(f"Unsupport datatype {self.data_type}!")

    def _init_custom(self):
        pass

    @torch.no_grad()
    def forward(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        is_prefill=True,
    ):
        if is_prefill:
            return self._prefill(
                batch_size,
                total_token_num,
                max_len_in_batch,
                input_ids,
                b_req_idx,
                b_start_loc,
                b_seq_len
            )
        else:
            return self._decode(
                batch_size,
                total_token_num,
                max_len_in_batch,
                input_ids,
                b_req_idx,
                b_start_loc,
                b_seq_len,
            )

    def _prefill(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids,
        b_req_idx,
        b_start_loc,
        b_seq_len
        # b_ready_cache_len,
    ):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0]
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        # zeyu: using to recover paused prefill and splitfuse
        # if b_ready_cache_len is not None:
            # infer_state.b_ready_cache_len = b_ready_cache_len
        # else:
        infer_state.b_ready_cache_len = torch.zeros_like(b_seq_len, dtype=b_seq_len.dtype, device=b_seq_len.device)

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_mem = self.mem_manager.alloc_contiguous(input_ids.shape[0])
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]

        else:
            infer_state.mem_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(input_ids.shape[0])
            infer_state.mem_index = alloc_mem
            infer_state.kv_buffer = torch.empty(
                (input_ids.shape[0], self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_),
                dtype=self.data_type,
                device="cuda",
            )

        init_req_to_token_indexes(
            self.req_manager.req_to_token_indexs,
            b_req_idx,
            b_seq_len,
            infer_state.b_ready_cache_len,
            max_len_in_batch,
            infer_state.mem_index,
        )

        infer_state.init_some_extra_state(self, input_ids)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics

    def _decode(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids,
        b_req_idx,
        b_start_loc,
        b_seq_len
    ):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0]
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        # 在使用 cuda graph 特性的时候，必须保证每次推理的流程一致
        # 所以不再使用分配连续的mem带来的优化，保证推理流程的一致
        alloc_mem = self.mem_manager.alloc_contiguous(batch_size)
        infer_state.mem_is_contiguous = True
        infer_state.mem_index = alloc_mem[0]
        infer_state.mem_start = alloc_mem[1]
        infer_state.mem_end = alloc_mem[2]
        copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)
        infer_state.init_some_extra_state(self, input_ids)
        predict_logics = self._token_forward(input_ids, infer_state)
        return predict_logics

    def _context_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(0, self.layers_num):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight)
        return predict_logics

    def _token_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(0, self.layers_num):
            input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight)
        return predict_logics