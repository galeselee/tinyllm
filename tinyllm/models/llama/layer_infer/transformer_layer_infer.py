import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from tinyllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from tinyllm.models.llama.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
)
from tinyllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from tinyllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from tinyllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2
from tinyllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from tinyllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from tinyllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd

from tinyllm.models.llama.infer_struct import LlamaInferStateInfo
from tinyllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv
from tinyllm.common.basemodel import TransformerLayerInferTpl


class LlamaTransformerLayerInfer(TransformerLayerInferTpl):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self._bind_func()
        return

    def _bind_func(self):
        self._bind_norm()
        self._bind_attention()
        return

    def _bind_norm(self):
        self._att_norm = partial(LlamaTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(LlamaTransformerLayerInfer._ffn_norm, self)
        return

    def _bind_attention(self):
        self._context_attention_kernel = partial(LlamaTransformerLayerInfer._context_attention_kernel, self)
        if "triton_flashdecoding" in self.mode:
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_flashdecoding, self
            )
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        elif "triton_gqa_attention" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_gqa_attention_normal, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        elif "triton_gqa_flashdecoding" in self.mode:
            self._token_attention_kernel = partial(
                LlamaTransformerLayerInfer._token_decode_attention_gqa_flashdecoding, self
            )
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        else:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_normal, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)

        return

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        rmsnorm_forward(input, weight=layer_weight.att_norm_weight_, eps=self.eps_, out=out)
        return out

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_, eps=self.eps_, out=out)
        return out

    def _get_qkv(
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        q = self.alloc_tensor((input.size(0), layer_weight.q_weight_.size(1)), data_type=input.dtype)
        torch.mm(input, layer_weight.q_weight_, out=q)
        torch.mm(
            input,
            layer_weight.kv_weight_,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        context_attention_fwd_no_prompt_cache(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv[:, 0 : self.tp_k_head_num_, :],
            kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )
        return o_tensor


    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = self.alloc_tensor((input.size(0), layer_weight.o_weight_.size(1)), input.dtype)
        torch.mm(input, layer_weight.o_weight_, out=o_tensor)
        return o_tensor

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        up_gate_out = self.alloc_tensor((input.size(0), layer_weight.gate_up_proj.size(1)), input.dtype)
        torch.mm(input, layer_weight.gate_up_proj, out=up_gate_out)
        ffn1_out = silu_and_mul_fwd(up_gate_out)
        input = None
        up_gate_out = None
        ffn2_out = self.alloc_tensor((ffn1_out.size(0), layer_weight.down_proj.size(1)), ffn1_out.dtype)
        torch.mm(ffn1_out, layer_weight.down_proj, out=ffn2_out)
        ffn1_out = None
        return ffn2_out

    # # keep code
    # def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight)->torch.Tensor:
    #     gate_up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_up_proj)
    #     size = gate_up_out.shape[1]
    #     gate_out, up_out = gate_up_out[:, 0: size // 2], gate_up_out[:, size // 2:]
    #     torch.nn.functional.silu(gate_out, inplace=True)
    #     gate_out.mul_(up_out)
    #     input = None
    #     ffn2_out = torch.mm(gate_out, layer_weight.down_proj)
    #     gate_out, up_out = None, None
    #     return ffn2_out

    def _copy_kv_to_mem_cache_normal(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(buffer, mem_index, mem_manager.kv_buffer[self.layer_num_])
        return


    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)

        att_m_tensor = self.alloc_tensor((self.tp_q_head_num_, total_token_num), torch.float32)

        token_att_fwd(
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            att_m_tensor,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )

        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        from tinyllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import (
            token_softmax_reducev_fwd,
        )

        token_softmax_reducev_fwd(
            att_m_tensor,
            infer_state.mem_manager.kv_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            o_tensor.view(calcu_shape1),
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.other_kv_index,
        )
        return o_tensor

    def _token_decode_gqa_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        # 对 gqa模型进行推理优化的代码
        from ..triton_kernel.gqa_decode_flashattention_nopad import gqa_decode_attention_fwd

        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        gqa_decode_attention_fwd(
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            infer_state.mem_manager.kv_buffer[self.layer_num_][
                :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
            ],
            o_tensor.view(calcu_shape1),
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
        )
        return o_tensor

    def _token_decode_attention_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        from tinyllm.models.llama.triton_kernel.flash_decoding import token_decode_attention_flash_decoding

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        return token_decode_attention_flash_decoding(
            q, infer_state, self.tp_q_head_num_, self.head_dim_, cache_k, cache_v, out=out
        )

    def _token_decode_attention_gqa_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        # 对 gqa 模型进行推理优化的代码
        from ..triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        return gqa_token_decode_attention_flash_decoding(
            q, infer_state, self.tp_q_head_num_, self.head_dim_, cache_k, cache_v, out=out
        )
    