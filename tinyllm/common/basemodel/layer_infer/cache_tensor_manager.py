import os
import torch
import collections
import dataclasses
import numpy as np
import torch._C
from typing import Dict, Iterable, Literal, Tuple, Union, List, Set
from torch.storage import UntypedStorage
from dataclasses import field
from tinyllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class CacheTensorManager:
    def __init__(self, cache_dir: str = None):
        pass

    def cache_env_in(self, is_cuda_graph: bool = False,
                     cur_batch_size: int = 0,
                     cuda_graph_max_batch_size: int = 0):
        return 
    
    def cache_env_out(self):
        return 

    def alloc_tensor(self, shape: Union[torch.Size, Iterable[int]], data_type: torch.dtype, device: str = "cuda") -> torch.Tensor:
        return 
    
    