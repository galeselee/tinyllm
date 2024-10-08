import torch 
from typing import Dict, Iterable, Literal, Tuple, Union, List
from tinyllm.common.basemodel.infer_struct import LayerInferState
from tinyllm.common.basemodel.layer_infer.base_layer_infer import BaseLayerInfer

class BaseLayerInfer:
    def __init__(self) -> None:
        pass

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def alloc_tensor(self, shape: Union[torch.Size, Iterable[int]], data_type: torch.dtype, device: str = "cuda")->torch.Tensor:
        return torch.empty(shape, dtype=data_type, device=device)

        
