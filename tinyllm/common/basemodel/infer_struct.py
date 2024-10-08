import torch
from tinyllm.common.managers.mem_manager import MemoryManager
from tinyllm.common.managers.req_manager import ReqManager


class InferStateInfo:
    """
    推理时用的信息结构体
    """

    def __init__(self):
        self.batch_size = None
        self.total_token_num = None
        self.b_req_idx = None
        self.b_start_loc = None
        self.b_ready_cache_len = None  # only for prefill prompt cache used.
        self.b_seq_len = None
        self.max_len_in_batch = None
        self.is_prefill = None

        self.mem_manager: MemoryManager = None
        self.req_manager: ReqManager = None

        self.mem_is_contiguous = None
        self.mem_index = None
        self.mem_start = None
        self.mem_end = None
        self.kv_buffer = None

        self.is_token_healing = False
        self.return_all_prompt_logics = False

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        pass
