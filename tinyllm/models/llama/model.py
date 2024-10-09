import os
import json
import torch
import math
from tinyllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from tinyllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from tinyllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from tinyllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from tinyllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from tinyllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

from tinyllm.models.llama.infer_struct import LlamaInferStateInfo
from tinyllm.common.basemodel import TpPartBaseModel
from tinyllm.common.managers.mem_manager import MemoryManager
from tinyllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class LlamaTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = LlamaPreAndPostLayerWeight
    transformer_weight_class = LlamaTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = LlamaTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        self._reset_num_key_value_heads()
        return

    def _reset_num_key_value_heads(self):
        if "num_key_value_heads" not in self.config:
            self.config["num_key_value_heads"] = self.config["num_attention_heads"]
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "llama only supports HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    def _init_mem_manager(self):
        self.mem_manager = MemoryManager(self.mode)(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_key_value_heads"] // self.world_size_,
            head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
            layer_num=self.config["num_hidden_layers"],
        )
        return

    def _init_custom(self):
        """
        模型特殊的一些初始化
        """
        if (
            self.config.get("rope_scaling", None) is not None
            and self.config.get("rope_scaling", {}).get("rope_type", "base") == "llama3"
        ):
            self._init_to_get_llama3_rotary()
        else:
            self._init_to_get_rotary()
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

    def _init_to_get_rotary(self, default_base=10000):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_)
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings", 2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        # NTK
        try:
            ntk_alpha = float(os.environ.get("TINYLLM_NTK_ALPHA", 1))
            assert ntk_alpha >= 1
            if ntk_alpha > 1:
                logger.info(f"Note: NTK enabled, alpha set to {ntk_alpha}")
            max_seq_len *= ntk_alpha
            base = base * (ntk_alpha ** (partial_head_dim / (partial_head_dim - 2)))  # Base change formula
        except:
            pass

        inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        t = (
            torch.arange(max(max_seq_len + 1024 * 128, self.max_seq_length), device="cpu", dtype=torch.float32)
            / rope_scaling_factor
        )
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return

    def _init_to_get_llama3_rotary(self, default_base=10000):
        partial_head_dim = int(self.config.get("partial_rotary_factor", 1) * self.head_dim_)
        base = self.config.get("rope_theta", float(default_base))

        scale_factor = self.config.get("rope_scaling", {}).get("factor", 8.0)
        low_freq_factor = self.config.get("rope_scaling", {}).get("low_freq_factor", 1.0)
        high_freq_factor = self.config.get("rope_scaling", {}).get("high_freq_factor", 4.0)
        origin_context_len = self.config.get("rope_scaling", {}).get("original_max_position_embeddings", 8192)

        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        max_seq_len = max_position_embeddings

        inv_freq = 1.0 / (
            base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )

        low_freq_wavelen = origin_context_len / low_freq_factor
        high_freq_wavelen = origin_context_len / high_freq_factor
        new_inv_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_inv_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_inv_freqs.append(freq / scale_factor)
            else:
                smooth = (origin_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_inv_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        inv_freq = torch.tensor(new_inv_freqs, dtype=torch.float32, device="cpu")

        t = torch.arange(max(max_seq_len, self.max_seq_length), device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(self.data_type).cuda()
        self._sin_cached = torch.sin(freqs).to(self.data_type).cuda()
        return
