import os
import asyncio
import numpy as np
import rpyc
import torch
from datetime import timedelta
from typing import Dict, List, Tuple
from tinyllm.models.llama.model import LlamaTpPartModel
from tinyllm.utils.infer_utils import set_random_seed
from tinyllm.utils.log_utils import init_logger

from tinyllm.common.infer_batch import InferBatch, InferReq, requests_mapping
from transformers.configuration_utils import PretrainedConfig


class ModeBackend:
    def __init__(self) -> None:
        pass

    def init_model(self, kvargs):
        import torch
        import torch.distributed as dist

        world_size = kvargs["world_size"]
        self.args = kvargs.get("args", None)
        self.is_multimodal = False
        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.eos_id: List[int] = kvargs.get("eos_id", [2])

        self.cache = {}
        self.logger = init_logger(__name__)

        self.weight_dir = kvargs["weight_dir"]
        max_total_token_num = kvargs["max_total_token_num"]

        torch.cuda.set_device(self.tp_rank)

        model_cfg, _ = PretrainedConfig.get_config_dict(self.weight_dir)

        model_kvargs = {
            "tp_rank": self.tp_rank,
            "world_size": self.world_size,
            "weight_dir": self.weight_dir,
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "data_type": kvargs.get("data_type", "float16"),
        }

        try:
            self.model_type = model_cfg.get("model_type", "")
            self.model = LlamaTpPartModel(model_kvargs)
        except Exception as e:
            self.logger.error(f"load model error: {str(e)} {e} {type(e)}")
            import traceback

            traceback.print_exc()
            raise e

        set_random_seed(2147483647)

        self.logger.info(f"loaded model class {self.model.__class__}")
        self.init_custom()

        return

    def init_custom(self):
        pass

    # @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        raise NotImplementedError()

    # @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        raise NotImplementedError()

    # @calculate_time(show=True, min_cost_ms=0.1)
    def add_batch(self, batch_id, reqs):
        batch_data = InferBatch.init_batch(
            batch_id,
            reqs,
            self.model.data_type,
            torch.cuda.current_device(),
            self.model.req_manager,
            self.model.vocab_size,
            self.radix_cache,
        )
        self.cache[batch_id] = batch_data

        # 将更新后的状态返回给调用方用于router中请求的状态
        ans = {}
        for req_id in batch_data.request_ids:
            req_obj: InferReq = requests_mapping[req_id]
            # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数
            ans[req_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [],
                req_obj.finish_status.value,
                None,
            )
        return ans

    # @calculate_time(show=True, min_cost_ms=0.1)
    def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    def pause_reqs(self, batch_id, req_list):
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.pause_reqs(req_list)
        self.cache[batch_id] = batch2
        del batch1
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def remove_batch(self, batch_id):
        batch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        return
