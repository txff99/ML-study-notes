from tensor import Op, OpType, Tensor
import numpy as np
from typing import List
import ctypes

class Device:
    def __init__(self):
        self.ops = None
        self.name = "dummy"

    def run(self):
        pass

class CPU(Device):
    def __init__(self):
        super().__init__()
        self.name = "cpu"
    
    def apply(self, ops: List[Op] = None):
        self.ops = ops
    
    def run(self):
        assert self.ops is not None , "ops should be applied before run"
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


class GPU(Device):
    def __init__(self, libpath: str):
        super().__init__()
        self.name = "gpu"
        self.lib = ctypes.CDLL(libpath)
        self.lib.alloc_on_gpu.argtypes = [ctypes.c_int]
        self.lib.alloc_on_gpu.restype = ctypes.c_void_p
        self.lib.free_on_gpu.argtypes = [ctypes.c_void_p]
        self.lib.free_on_gpu.restype = ctypes.c_void_p
        self.lib.cuda_copy_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuda_copy_to_device.restype = ctypes.c_void_p
        self.lib.cuda_copy_to_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuda_copy_to_host.restype = ctypes.c_void_p
        
        self.lib.matmul.argtypes = [
            ctypes.c_void_p,  # const float* A
            ctypes.c_void_p,  # const float* B
            ctypes.c_void_p,  # float* C
            ctypes.c_int,     # int M
            ctypes.c_int,     # int N
            ctypes.c_int      # int K
        ]
        self.lib.matmul.restype = ctypes.c_void_p
    
    def apply(self, ops: List[Op] = None):
        self.ops = ops
    
    def alloc(self, tensor: Tensor):
        # alloc the mem for each tensor in gpu and give back a ptr to the tensor 
        assert tensor.device.name == "gpu", "tensor should be on gpu"
        size = np.prod(tensor.shape) * tensor.data.itemsize
        tensor.device_ptr = self.lib.alloc_on_gpu(size)

    def copyHTOD(self, tensor: Tensor):
        assert tensor.device.name == "gpu", "tensor should be on gpu"
        assert tensor.device_ptr is not None, "tensor device pointer not found"
        # tensor.data = np.ascontiguousarray(tensor.data)
        ptr = tensor.data.ctypes.data
        self.lib.cuda_copy_to_device(tensor.device_ptr, ptr, tensor.data.nbytes)
    
    def copyDTOH(self, tensor: Tensor):
        assert tensor.device.name == "gpu", "tensor should be on gpu"
        assert tensor.device_ptr is not None, "tensor device pointer not found"
        if tensor.is_evaluated == False: tensor.data = np.zeros(tensor.shape, dtype=tensor.dtype)
        ptr = tensor.data.ctypes.data
        size = np.prod(tensor.shape) * tensor.data.itemsize
        self.lib.cuda_copy_to_host(ptr, tensor.device_ptr, size)

    def free(self, tensor: Tensor):
        assert tensor.device.name == "gpu", "tensor should be on gpu"
        assert tensor.device_ptr is not None, "tensor dont have device ptr"
        self.lib.free_on_gpu(tensor.device_ptr)
        tensor.device_ptr = None

    def run(self):
        assert self.ops is not None , "ops should be applied before run"
        for op in self.ops:
            optype: OpType = op.optype
            srcs: List[Tensor] = op.srcs
            dst: Tensor = op.dst
            if dst.is_evaluated == True: continue
            # optype matching
            if optype == OpType.MATMUL:
                assert len(srcs) == 2, "srcs num does not match"
                assert srcs[0].device_ptr is not None and srcs[1].device_ptr is not None, "srcs dont have device ptr"
                assert dst.device_ptr is not None, "dst doesnt have device ptr"
                self.lib.matmul(srcs[0].device_ptr,
                                srcs[1].device_ptr, 
                                dst.device_ptr, 
                    srcs[0].shape[0], srcs[1].shape[1], srcs[0].shape[1])
            else:
                raise NotImplementedError(f"Operation '{optype}' is not implemented.")