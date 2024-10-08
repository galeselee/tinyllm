# import faulthandler
# faulthandler.enable()

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory


class SharedArray:
    def __init__(self, name, shape, dtype):
        dtype_byte_num = np.array([1], dtype=dtype).dtype.itemsize
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=np.prod(shape) * dtype_byte_num)
            print(f"create shm {name}")
        except:
            shm = shared_memory.SharedMemory(name=name, create=False, size=np.prod(shape) * dtype_byte_num)
            assert (
                len(shm.buf) == np.prod(shape) * dtype_byte_num
            ), f"{len(shm.buf)} is not equal to {np.prod(shape) * dtype_byte_num}"
            print(f"link shm {name}")
        self.shm = shm  # SharedMemory 对象一定要被持有，否则会被释放
        self.arr = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)


class SharedInt(SharedArray):
    def __init__(self, name):
        super().__init__(name, shape=(1,), dtype=np.int64)

    def set_value(self, value):
        self.arr[0] = value

    def get_value(self):
        return self.arr[0]