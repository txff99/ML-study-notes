from __future__ import annotations
import numpy as np
from typing import List
from enum import Enum
from util import get_default_strides

"""
tensor used to provide interface for data operation
"""

class OpType(Enum):
    ADD = 1
    SUB = 2
    MATMUL = 3
    MSELOSS = 4 
    MAXIMUM = 5
    MAX = 6
    MUL = 7
    DIV = 8
    SUM = 9
    EXP = 10
    SQRT = 11
    EXPAND = 12
    TRANSPOSE = 13
    CONTIGUOUS = 14

FUNCTION_TO_OPTYPE = {
    "add": OpType.ADD,
    "sub": OpType.SUB,
    "matmul": OpType.MATMUL,
    "mse": OpType.MSELOSS,
    "maximum": OpType.MAXIMUM,
    "max": OpType.MAX,
    "mul": OpType.MUL,
    "div": OpType.DIV,
    "exp": OpType.EXP,
    "sum": OpType.SUM,
    "sqrt": OpType.SQRT,
    "expand": OpType.EXPAND,
    "transpose": OpType.TRANSPOSE,
    "contiguous": OpType.CONTIGUOUS
}

class Op:
    def __init__(self, optype: OpType, srcs: List[Tensor], dst: Tensor):
        self.optype = optype
        self.srcs = srcs
        self.dst = dst

    def __repr__(self):
        return f"{self.optype.name}({', '.join(str(id(s))[-4:] for s in self.srcs)} -> {str(id(self.dst))[-4:]})"

class Tensor:
    def __init__(self, data=None, function=None, shape: tuple|list=None, 
                    dtype=np.float32, strides:tuple|list = None, 
                    is_realized=True):
        from backend import CPU
        self.data = np.asarray(data, dtype=dtype)
        self.dtype = dtype
        self.gradient: np.array = None
        self.function = function
        self._shape = list(shape) if shape is not None else self.data.shape 
        self._strides = list(strides) if strides is not None else None
        self.backend = CPU()
        self.backend_ptr = None
        self.is_realized = is_realized
    
    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def strides(self):
        return tuple(self._strides) if self._strides is not None else get_default_strides(self.shape)

    def numpy(self) -> np.array:
        contiguous_tensor = self.contiguous()
        if not contiguous_tensor.is_realized:
            contiguous_tensor.realize()
        return contiguous_tensor.data

    def lower(self) -> List(Op): 
        # lower a dag to ops
        visited = set()
        ops = []
        def dfs(tensor: Tensor):
            if tensor.function is None: return
            if tensor in visited: return
            visited.add(tensor)    
            for mem in tensor.function.parents:
                dfs(mem)
            ops.append(Op(FUNCTION_TO_OPTYPE[tensor.function.name], tensor.function.parents, tensor))
        dfs(self)
        return ops
    
    def toGPU(self):
        """allocate mem on gpu for itself and its parents"""
        from backend import GPU
        libpath = "/home/lawlet/Documents/tutorial/dl_impl/runtime/cuda/libcuda.so"
        # cpu does not require resource dealloc, so replace it with gpu directly
        self.backend = GPU(libpath)
        visited = set()
        def dfs(tensor: Tensor):
            if tensor in visited: return
            tensor.backend = self.backend
            visited.add(tensor)
            self.backend.alloc(tensor)
            if tensor.is_realized == True:
                self.backend.copyHTOD(tensor)
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
        dfs(self)        

    def toCPU(self, cleanup: bool = False):
        """copy gpu data back to cpu; clean up (optional)"""
        from backend import CPU
        if self.backend.name != "gpu": return
        visited = set()
        def dfs(tensor: Tensor):
            if tensor in visited: return
            visited.add(tensor)    
            if tensor.is_realized == False:
                # only set is_realized when copy the data to host
                self.backend.copyDTOH(tensor)
                tensor.is_realized = True
            if cleanup:
                self.backend.free(tensor)
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
        dfs(self)        
        self.backend = CPU()

    def realize(self, cleanup_after_eval=True):
        from engine import Engine
        engine = Engine(self.backend, self.lower())
        engine.run()
        if self.backend.name == "gpu":
            # gpu need cleanup
            self.toCPU(cleanup=cleanup_after_eval)

    def __add__(self, other: Tensor) -> Tensor:
        from function import Add
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(None, function=Add(self,other), shape=other.shape ,is_realized=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def __sub__(self, other:Tensor) -> Tensor:
        from function import Sub
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(None,function=Sub(self,other), shape=other.shape,is_realized=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def __mul__(self, other:Tensor) -> Tensor:
        from function import Mul
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(None,function=Mul(self,other), shape=other.shape,is_realized=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")

    def __div__(self, other:Tensor) -> Tensor:
        from function import Div
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(None,function=Div(self,other), shape=other.shape,is_realized=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")

    def __matmul__(self, other: Tensor) -> Tensor:
        from function import MatMul
        if isinstance(other, Tensor):
            # Ensure that matrix dimensions are compatible for multiplication
            if self.data.shape[-1] != other.data.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {self.data.shape} and {other.data.shape}")
            return Tensor(None,function=MatMul(self,other), shape=(self.data.shape[0], other.data.shape[1]), is_realized=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def expand(self, *expanded_shape: tuple[uint8]) -> Tensor:
        """
        expand only return a view
        """
        from function import Expand
        assert len(self.shape) == len(expanded_shape), \
            "expanded shape should have same dimensions as source shape"
        assert all(s == e or s == 1 for s, e in zip(self.shape, expanded_shape)), \
            "expand should perform on singleton dimension"
        strides = self.strides
        if strides is None:
            strides = get_default_strides(self.shape)
        for i, dim in enumerate(self.shape):
            if dim == 1 and expanded_shape[i] != self.shape[i]:
                strides[i] = 0
        
        return Tensor(None,function=Expand(self,expanded_shape),shape=expanded_shape,strides=strides, is_realized=False)

    def transpose(self, dim1, dim2) -> Tensor:
        from function import Transpose
        assert dim1 < len(self.shape) and dim2 < len(self.shape), "dimension out of bound"
        new_shape = list(self.shape)
        new_shape[dim1], new_shape[dim2] = self.shape[dim2], self.shape[dim1]
        new_strides = list(self.strides) if self.strides is not None else get_default_strides(new_shape)
        new_strides[dim1], new_strides[dim2] = new_strides[dim2], new_strides[dim1]
        return Tensor(None, function=Transpose(self,dim1,dim2),shape=new_shape,strides=new_strides, is_realized=False)

    def is_contiguous(self):
        return (self._strides is None or (self._strides == get_default_strides(self.shape)))

    def contiguous(self) -> Tensor:
        """ 
        (1,3,1) -> (3,3,1) stride (0,1,1)
        (3,1,1)
        for ptr in range(np.prod(shape)):
            indices = np.zeros(len(shape))
            rem = ptr
            # compute index for ptr
            for i,s in enumerate(get_default_stride(shape)):
                indices[i] = rem // s
                rem%=s
            old_data_ptr = np.dot(indices, strides)
            data[ptr] = old_data[old_data_ptr]
        map the new tensor's data using the transformed strides
        """
        from function import Contiguous
        if self.is_contiguous():
            return self
        return Tensor(None,function=Contiguous(self),shape=self.shape,is_realized=False)

    def unsqueeze(self, dim: int) -> Tensor:
        pass

    def maximum(self, gate: int) -> Tensor:
        from function import Maximum
        return Tensor(None, function=Maximum(self, gate), shape=self.shape, is_realized=False)

    def exp(self) -> Tensor:
        from function import Exp
        return Tensor(None, functon=Exp(self), shape=self.shape, is_realized=False)

    def sqrt(self) -> Tensor:
        from function import Sqrt
        return Tensor(None, function=Sqrt(self), shape=self.shape, is_realized=False)

    def max(self, dim:int) -> Tensor:
        from function import Max
        return Tensor(None, function=Max(self, dim), shape=self.shape[:dim] + self.shape[dim+1:], is_realized=False)
    
    def sum(self, dim:int) -> Tensor:
        from function import Sum
        return Tensor(None, function=Sum(self, dim), shape=self.shape[:dim] + self.shape[dim+1:], is_realized=False)
    
    def mseLoss(self) -> Tensor:
        from function import MSELoss
        return Tensor(None,function=MSELoss(self), shape=(1,),is_realized=False)

    def concate(self, dim:int) -> Tensor:
        pass
    
    def __repr__(self):
        return f"Tensor(data= {self.data}, shape={self.shape}, grad={self.gradient}, function: {self.function}, strides: {self.strides}, is_realized: {self.is_realized})"
    
    def backward(self, level:str = None):
        if self.function is None: return
        if self.gradient is None: 
            self.gradient = np.array(1)
            level = ""
        if self.is_realized:
            self.function.res = self.data
        assert self.gradient.shape == self.shape, "gradient's shape should equal to tensor shape"
        self.function.backward(self.gradient)
