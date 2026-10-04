from __future__ import annotations
import numpy as np
from typing import List
from enum import Enum
from util import get_default_strides, canonicalize_index, get_new_shape_from_index

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
    INDEXGET = 15
    INDEXPUT = 16
    RESHAPE = 17

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
    "contiguous": OpType.CONTIGUOUS,
    "index_get": OpType.INDEXGET,
    "index_put": OpType.INDEXPUT,
    "reshape": OpType.RESHAPE,
}

class Op:
    def __init__(self, optype: OpType, srcs: List[Tensor], dst: Tensor):
        self.optype = optype
        self.srcs = srcs
        self.dst = dst

    def __repr__(self):
        return f"{self.optype.name}({', '.join(str(id(s))[-4:] for s in self.srcs)} -> {str(id(self.dst))[-4:]})"

class Tensor:
    def __init__(self, data:np.array|tuple|list|int|float=None, function=None, shape: tuple|list=None, 
                    dtype=np.float32, strides:tuple|list = None, offset:int=None, require_grad=False, 
                    is_realized=True, is_fake=False):
        assert data is not None or shape is not None, "at least one of the data or shape should be given"
        if data is not None and shape is not None:
            np.prod(data.shape)==np.prod(shape), "data should match shape"
        from backend import CPU
        self.data = np.asarray(data, dtype=dtype).flatten() # data will always be flatten
        self.dtype = dtype
        self.gradient: np.array = None
        self.function = function
        self._shape = list(shape) if shape is not None else list(np.asarray(data).shape)
        self._strides = list(strides) if strides is not None else None
        self._offset = offset
        self.backend = CPU()
        self.backend_ptr = None
        self.is_realized = is_realized
        self.require_grad=require_grad
        self.is_fake=is_fake
    
    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def strides(self):
        return tuple(self._strides) if self._strides is not None else get_default_strides(self.shape)

    @property
    def offset(self):
        return self._offset
    
    @property
    def is_scalar(self):
        return len(self.shape) == 0
    
    def numpy(self) -> np.array:
        contiguous_tensor = self.contiguous()
        if not contiguous_tensor.is_realized:
            contiguous_tensor.realize()
        if self.is_scalar: return contiguous_tensor.data
        return contiguous_tensor.data.reshape(self.shape)

    def lower(self) -> List(Op): 
        # lower a dag to ops
        from graph import Graph
        g = Graph(self)
        if self.backend.name == "cpu":
            from graph import add_contiguous_before_ari
            g.rewrite(add_contiguous_before_ari)
        tensors = g.toposort()
        ops = [Op(FUNCTION_TO_OPTYPE[tensor.function.name], tensor.function.parents, tensor) for tensor in tensors]
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
            if tensor.function is None: return
            tensor.backend = self.backend
            visited.add(tensor)
            self.backend.alloc(tensor)
            if tensor.is_realized == True:
                self.backend.copyHTOD(tensor)
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
            if tensor.function is None: return
            if tensor.is_realized == False:
                # only set is_realized when copy the data to host
                self.backend.copyDTOH(tensor)
                tensor.is_realized = True
            if cleanup:
                self.backend.free(tensor)
            for mem in tensor.function.parents:
                dfs(mem)
        dfs(self)        
        self.backend = CPU()

    def realize(self, backend=None, cleanup_after_eval=True, plan: List[Op]=None):
        from engine import Engine
        if plan is None:
            plan = self.lower()
        backend=self.backend if backend is None else backend
        engine = Engine(backend, plan)
        engine.run()
        if backend.name == "gpu":
            # gpu need cleanup
            self.toCPU(cleanup=cleanup_after_eval)

    def __add__(self, other: Tensor) -> Tensor:
        from function import Add
        assert self.shape==other.shape , "tensor shape is not the same"
        assert isinstance(other, Tensor), "The operand must be an instance of Tensor"
        return Tensor(None, function=Add(self,other), shape=other.shape ,is_realized=False)
    
    def __sub__(self, other:Tensor) -> Tensor:
        from function import Sub
        assert self.shape==other.shape , "tensor shape is not the same"
        assert isinstance(other, Tensor), "The operand must be an instance of Tensor"
        return Tensor(None,function=Sub(self,other), shape=other.shape,is_realized=False)
    
    def __mul__(self, other:Tensor) -> Tensor:
        from function import Mul
        assert self.shape==other.shape , "tensor shape is not the same"
        assert isinstance(other, Tensor), "The operand must be an instance of Tensor"
        return Tensor(None,function=Mul(self,other), shape=other.shape,is_realized=False)

    def __truediv__(self, other:Tensor) -> Tensor:
        from function import Div
        assert self.shape==other.shape , "tensor shape is not the same"
        assert isinstance(other, Tensor), "The operand must be an instance of Tensor"
        return Tensor(None,function=Div(self,other), shape=other.shape,is_realized=False)

    def __matmul__(self, other: Tensor) -> Tensor:
        from function import MatMul
        assert isinstance(other, Tensor), "The operand must be an instance of Tensor"
        assert len(self.shape) == 2 and len(other.shape)==2, "matmul only support 2 dim as input"
        assert self.shape[-1] == other.shape[0], f"Incompatible shapes for matrix multiplication: {self.shape} and {other.shape}"
        return Tensor(None,function=MatMul(self,other), shape=(self.shape[0], other.shape[1]), is_realized=False)
    
    def expand(self, *expanded_shape: tuple[int]) -> Tensor:
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

    def reshape(self, *new_shape: tuple[int]) -> Tensor:
        from function import Reshape
        assert np.prod(new_shape) == np.prod(self.shape), "element num does not match"
        new_strides = get_default_strides(new_shape)
        return Tensor(None, function=Reshape(self, new_shape), shape=new_shape,strides=new_strides, offset=self.offset, is_realized=False)
    
    def __getitem__(self, key) -> Tensor:
        assert not self.is_scalar, "tensor is a scalar and should not be indexed."
        from function import IndexGet
        key = canonicalize_index(key, self.shape)
        new_shape, new_strides, offset = get_new_shape_from_index(
            key,
            self.shape,
            self.strides,
            self.offset,
        )
        return Tensor(None, function=IndexGet(self,key),shape=new_shape,strides=new_strides,offset=offset,is_realized=False)
                
    def __setitem__(self, key, value) -> Tensor:
        old_self = Tensor(
            data=self.data,
            function=self.function,
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
            is_realized=self.is_realized
        )
        new = old_self.index_put(key, value)
        self.__dict__ = new.__dict__

    def index_put(self, key, value) -> Tensor:
        assert not self.is_scalar, "tensor is a scalar and should not be indexed."
        assert isinstance(value, Tensor), "index put should have value as tensor."
        from function import IndexPut
        key = canonicalize_index(key, self.shape)
        new_shape, new_strides, offset = get_new_shape_from_index(
            key,
            self.shape,
            self.strides,
            self.offset,
        )
        assert tuple(new_shape) == value.shape, f"value shape does not match key shape. {new_shape}, {value.shape}"
        shape_info = (new_shape,new_strides,offset)
        return Tensor(None,function=IndexPut(self,shape_info,value),shape=self.shape,strides=self.strides,offset=self.offset,is_realized=False)

    def is_contiguous(self):
        return (self.offset is None and (self._strides is None or (self._strides == get_default_strides(self.shape))))

    def contiguous(self) -> Tensor:
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
        return Tensor(None, function=Exp(self), shape=self.shape, is_realized=False)

    def sqrt(self) -> Tensor:
        from function import Sqrt
        return Tensor(None, function=Sqrt(self), shape=self.shape, is_realized=False)

    def max(self, dim:int) -> Tensor:
        assert dim < len(self.shape)
        from function import Max
        return Tensor(None, function=Max(self, dim), shape=self.shape[:dim] + self.shape[dim+1:], is_realized=False)
    
    def sum(self, dim:int) -> Tensor:
        assert dim < len(self.shape)
        from function import Sum
        return Tensor(None, function=Sum(self, dim), shape=self.shape[:dim] + self.shape[dim+1:], is_realized=False)
    
    def mseLoss(self) -> Tensor:
        from function import MSELoss
        return Tensor(None,function=MSELoss(self), shape=(1,),is_realized=False)

    def concate(self, dim:int) -> Tensor:
        pass
    
    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, grad={self.gradient}, function={self.function}, strides={self.strides}, is_realized={self.is_realized})"
    
    def replace(self, other: Tensor):
        self.__dict__.update(other.__dict__)

    def makeFakeEdge(self, other: Tensor) -> Tensor:
        from function import FakeFunction
        return Tensor(None, shape=self.shape, function=FakeFunction(self, other), is_realized=False)
    
    def backward(self, level:str = None):
        if self.function is None: return
        if self.gradient is None: 
            self.gradient = np.array(1)
            level = ""
        if self.is_realized:
            self.function.res = self.data
        assert self.gradient.shape == self.shape, "gradient's shape should equal to tensor shape"
        self.function.backward(self.gradient)
