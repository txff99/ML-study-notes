from tensor import Op, OpType, Tensor
import numpy as np
from typing import List
import ctypes

class Device:
    def __init__(self, ops: List[Op]=None):
        self.ops = ops

    def run(self):
        pass

class CPU(Device):
    def __init__(self, ops: List[Op]=None):
        super().__init__(ops)
    
    def run(self):
        for op in self.ops:
            optype: OpType = op.optype
            srcs: List[Tensor] = op.srcs
            dst: Tensor = op.dst
            if dst.is_evaluated == True: continue
            # optype matching
            if optype == OpType.ADD:
                assert len(srcs) == 2, "srcs num does not match"
                dst.data = srcs[0].data + srcs[1].data
            elif optype == OpType.SUB:
                assert len(srcs) == 2, "srcs num does not match"
                dst.data = srcs[0].data - srcs[1].data
            elif optype == OpType.MATMUL:
                assert len(srcs) == 2, "srcs num does not match"
                dst.data = srcs[0].data @ srcs[1].data
            elif optype == OpType.EXPAND:
                assert len(srcs) == 2, "srcs num does not match"
                dst.data = np.repeat(np.expand_dims(srcs[0].data,axis=0),srcs[1].data,axis=0)  
            elif optype == OpType.MSELOSS:
                assert len(srcs) == 1, "srcs num does not match"
                dst.data = np.mean(srcs[0].data**2)
            else:
                raise NotImplementedError(f"Operation '{optype}' is not implemented.")


class GPU:
    def __init__(self, ops: List[Op]=None):
        super().__init__(ops)
        self.lib = ctypes.CDLL('./libcuda.so')
        self.lib.alloc_on_gpu.argtypes = [ctypes.c_int]
        self.lib.alloc_on_gpu.restype = ctypes.c_void_p
        self.lib.free_on_gpu.argtypes = [ctypes.c_void_p]
        self.lib.free_on_gpu(tensor.device_ptr)
        self.lib.cuda_copy_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuda_copy_to_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.matmul.argtypes = [
            ctypes.c_void_p,  # const float* A
            ctypes.c_void_p,  # const float* B
            ctypes.c_void_p,  # float* C
            ctypes.c_int,     # int M
            ctypes.c_int,     # int N
            ctypes.c_int      # int K
        ]
        self.lib.matmul.restype = None
    
    def alloc(self, tensor: Tensor):
        # alloc the mem for each tensor in gpu and give back a ptr to the tensor 
        assert tensor.device == "gpu", "tensor should be on gpu"
        size = np.product(tensor.shape) * np.dtype(tensor.dtype).itemsize
        tensor.device_ptr = self.lib.alloc_on_gpu(size)

    def copyHTOD(self, tensor: Tensor):
        assert tensor.device_ptr is not None and tensor.is_evaluated == True, "tensor has not copied to gpu or is not evaluated"
        ptr = tensor.data.ctypes.data_as(ctypes.c_void_p)
        self.lib.cuda_copy_to_device(tensor.device_ptr, ptr, tensor.data.nbytes)
    
    def copyDTOH(self, tensor: Tensor):
        assert tensor.device_ptr is not None and tensor.is_evaluated == True, "tensor has not copied to gpu or is not evaluated"
        ptr = tensor.data.ctypes.data_as(ctypes.c_void_p)
        self.lib.cuda_copy_to_host(ptr, tensor.device_ptr, tensor.data.nbytes)

    def free(self, tensor: Tensor):
        assert tensor.device == "gpu", "tensor should be on gpu"
        assert tensor.device_ptr is not None, "tensor dont have device ptr"
        tensor.device_ptr = None

    def run(self):
        for op in self.ops:
            optype: OpType = op.optype
            srcs: List[Tensor] = op.srcs
            dst: Tensor = op.dst
            if dst.is_evaluated == True: continue
            # optype matching
            if optype == OpType.MATMUL:
                assert len(srcs) == 2, "srcs num does not match"
                self.lib.matmul(srcs[0].data, srcs[1].data, dst.data, 
                    srcs[0].data.shape[0], srcs[1].data.shape[1], srcs[0].data.shape[1])
            else:
                raise NotImplementedError(f"Operation '{optype}' is not implemented.")