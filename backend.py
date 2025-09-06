from tensor import Op, OpType, Tensor
import numpy as np
from typing import List
import ctypes

"""
backend specify memory allocation and op execution for each device
"""

class Backend:
    def __init__(self):
        self.name = "dummy"

    def execute(self, op: Op):
        pass

class CPU(Backend):
    def __init__(self):
        super().__init__()
        self.name = "cpu"
        self.supported_ops = {OpType.ADD, OpType.SUB, OpType.MATMUL, OpType.EXPAND, OpType.MSELOSS, OpType.MAX, OpType.TRANSPOSE, OpType.CONTIGUOUS}
    
    def execute(self, op: Op):
        optype: OpType = op.optype
        srcs: List[Tensor] = op.srcs
        dst: Tensor = op.dst
        if dst.is_realized == True: return
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
            assert len(srcs) == 1, "srcs num does not match"
            dst.data = np.broadcast_to(srcs[0].data, dst.shape)
        elif optype == OpType.TRANSPOSE:
            assert len(srcs) == 1, "srcs num does not match"
            dst.data = srcs[0].data
        elif optype == OpType.CONTIGUOUS:
            assert len(srcs) == 1, "srcs num does not match"
            dst.data = dst.function.contiguous_python_impl(srcs[0]).reshape(dst.shape)
        elif optype == OpType.MSELOSS:
            assert len(srcs) == 1, "srcs num does not match"
            dst.data = np.mean(srcs[0].data**2)
        elif optype == OpType.MAX:
            assert len(srcs) == 1, "srcs num does not match"
            gate = dst.function.gate
            mask = srcs[0].data > gate
            dst.data = srcs[0].data * mask
        else:
            raise NotImplementedError(f"Operation '{optype}' is not implemented.")


class GPU(Backend):
    def __init__(self, libpath: str):
        super().__init__()
        self.name = "gpu"
        self.supported_ops = {OpType.ADD, OpType.MATMUL, OpType.SUB, OpType.EXPAND}
        self.lib = ctypes.CDLL(libpath)
        self._define_func("alloc_on_gpu", [ctypes.c_int], ctypes.c_void_p)
        self._define_func("free_on_gpu", [ctypes.c_void_p], None)
        self._define_func("cuda_copy_to_device", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], None)
        self._define_func("cuda_copy_to_host", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t], None)
        self._define_func("matmul", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int], None)
        self._define_func("cutlass_mma", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int], None)
        self._define_func("add", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int], None)
        self._define_func("sub", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int], None)
        self._define_func("expand", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int], None)

    def _define_func(self, name, argtypes, restype):
        func = getattr(self.lib, name)
        func.argtypes = argtypes
        func.restype = restype if restype is not None else ctypes.c_void_p
    
    def apply(self, ops: List[Op] = None):
        self.ops = ops
    
    def alloc(self, tensor: Tensor):
        # alloc the mem for each tensor in gpu and give back a ptr to the tensor 
        assert tensor.backend.name == "gpu", "tensor should be on gpu"
        size = np.prod(tensor.shape) * tensor.data.itemsize
        tensor.backend_ptr = self.lib.alloc_on_gpu(size)

    def copyHTOD(self, tensor: Tensor):
        assert tensor.backend.name == "gpu", "tensor should be on gpu"
        assert tensor.backend_ptr is not None, "tensor backend pointer not found"
        # tensor.data = np.ascontiguousarray(tensor.data)
        ptr = tensor.data.ctypes.data
        self.lib.cuda_copy_to_device(tensor.backend_ptr, ptr, tensor.data.nbytes)
    
    def copyDTOH(self, tensor: Tensor):
        assert tensor.backend.name == "gpu", "tensor should be on gpu"
        assert tensor.backend_ptr is not None, "tensor backend pointer not found"
        if tensor.is_realized == False: tensor.data = np.zeros(tensor.shape, dtype=tensor.dtype)
        ptr = tensor.data.ctypes.data
        size = np.prod(tensor.shape) * tensor.data.itemsize
        self.lib.cuda_copy_to_host(ptr, tensor.backend_ptr, size)

    def free(self, tensor: Tensor):
        assert tensor.backend.name == "gpu", "tensor should be on gpu"
        assert tensor.backend_ptr is not None, "tensor doesnt have backend ptr"
        self.lib.free_on_gpu(tensor.backend_ptr)
        tensor.backend_ptr = None

    def _gpu_alloc_assert(self, srcs: List[Tensor], dst: Tensor):
        for src in srcs:
            assert src.backend_ptr is not None, "srcs dont have backend ptr"
        assert dst.backend_ptr is not None, "dst doesnt have backend ptr"

    def execute(self, op: Op):
        optype: OpType = op.optype
        srcs: List[Tensor] = op.srcs
        dst: Tensor = op.dst
        if dst.is_realized == True: return
        # optype matching
        if optype == OpType.MATMUL:
            assert len(srcs) == 2, "srcs num does not match"
            self.lib.matmul(srcs[0].backend_ptr,
                            srcs[1].backend_ptr, 
                            dst.backend_ptr, 
                srcs[0].shape[0], srcs[1].shape[1], srcs[0].shape[1])
        elif optype == OpType.ADD:
            assert len(srcs) == 2, "srcs num does not match"
            self._gpu_alloc_assert(srcs, dst)
            assert srcs[0].shape == srcs[1].shape and srcs[0].shape == dst.shape
            self.lib.add(srcs[0].backend_ptr,
                        srcs[1].backend_ptr, 
                        dst.backend_ptr, 
                srcs[0].shape[0], srcs[0].shape[1])
        elif optype == OpType.SUB:
            assert len(srcs) == 2, "srcs num does not match"
            self._gpu_alloc_assert(srcs, dst)
            assert srcs[0].shape == srcs[1].shape and srcs[0].shape == dst.shape
            self.lib.sub(srcs[0].backend_ptr,
                        srcs[1].backend_ptr, 
                        dst.backend_ptr, 
                srcs[0].shape[0], srcs[0].shape[1])
        elif optype == OpType.EXPAND:
            pass
        else:
            raise NotImplementedError(f"Operation '{optype}' is not implemented.")