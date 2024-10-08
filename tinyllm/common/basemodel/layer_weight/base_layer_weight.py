import torch
import numpy as np
import threading

class BaseLayerWeight:
    def __init__(self):
        self.tp_rank_ = None
        self.data_type_ = None
        self.lock = threading.Lock()
    
    def load_hf_weights(self, weights):
        pass

    def init_static_params(self):
        pass

    def verify_load(self):
        raise Exception("must verify weights load ok")
    
    def _cuda(self, cpu_tensor: torch.Tensor):
        assert self.data_type_ is not None
        if self.tp_rank_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).cuda()
        else:
            return cpu_tensor.contiguous().to(self.data_type_).cuda(self.tp_rank_) 
    
    def _try_cat_to(self, source_tensor_names, dest_name, cat_dim):
        if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
            assert all(not getattr(self, src_name, None).is_cuda for src_name in source_tensor_names), "all not cuda tensor"
            tensors = [getattr(self, src_name, None) for src_name in source_tensor_names]
            ans = torch.cat(tensors, dim=cat_dim) 
            ans = self._cuda(ans)
            setattr(self, dest_name, ans)
            for src_name in source_tensor_names:
                delattr(self, src_name)
                
        return
    
            
