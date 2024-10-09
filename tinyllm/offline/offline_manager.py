import copy
import time
import uuid
from typing import Dict, List, Optional
from tinyllm.common.shared_arr import SharedInt
from tinyllm.common.io_struct import Batch
from tinyllm.common.token_load import TokenLoad
from tinyllm.common.io_struct import Req
from tinyllm.utils.log_utils import logger
from tinyllm.offline.backend import ContinuesBatchBackend

class OfflineManager:
    def __init__(self, args, reqs: List[Req]):
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
        
        self.reqs = reqs
        self.reqs_num = len(reqs)

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
            