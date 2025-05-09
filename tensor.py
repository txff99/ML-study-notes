from __future__ import annotations
import numpy as np
from typing import List
from enum import Enum

"""
tensor used to provide interface for data operation
"""


class OpType(Enum):
    ADD = 1
    SUB = 2
    MATMUL = 3
    MSELOSS = 4 
    EXPAND = 5
    MAXIMUM = 6
    MAX = 7
    MUL = 8
    DIV = 9
    SUM = 10
    EXP = 11
    SQRT = 12

FUNCTION_TO_OPTYPE = {
    "add": OpType.ADD,
    "sub": OpType.SUB,
    "matmul": OpType.MATMUL,
    "mse": OpType.MSELOSS,
    "expand": OpType.EXPAND,
    "maximum": OpType.MAXIMUM,
    "max": OpType.MAX,
    "mul": OpType.MUL,
    "div": OpType.DIV,
    "exp": OpType.EXP,
    "sum": OpType.SUM,
    "sqrt": OpType.SQRT,
}

class Op:
    def __init__(self, optype: OpType, srcs: List[Tensor], dst: Tensor):
        self.optype = optype
        self.srcs = srcs
        self.dst = dst

    def __repr__(self):
        return f"{self.optype.name}({', '.join(str(id(s))[-4:] for s in self.srcs)} -> {str(id(self.dst))[-4:]})"

class Tensor:
    def __init__(self,data=None,function=None,shape=None,dtype=np.float32,is_evaluated=True):
        from backend import CPU
        self.data = np.asarray(data, dtype=dtype)
        self.dtype = dtype
        self.gradient: np.array = None
        self.function = function
        self.shape = shape if shape is not None else self.data.shape 
        self.is_evaluated = is_evaluated
        self.backend = CPU()
        self.backend_ptr = None
    
    def numpy(self) -> np.array:
        return self.data

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
            if tensor.is_evaluated == True:
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
            if tensor.is_evaluated == False:
                # only set is_evaluated when copy   the data to host
                self.backend.copyDTOH(tensor)
                tensor.is_evaluated = True
            if cleanup:
                self.backend.free(tensor)
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
        dfs(self)        
        self.backend = CPU()

    def evaluate(self, cleanup_after_eval=True) -> np.array:
        from engine import Engine
        engine = Engine(self.backend, self.lower())
        engine.run()
        if self.backend.name == "gpu":
            # gpu need cleanup
            self.toCPU(cleanup=cleanup_after_eval)
        return self.data

    def __add__(self, other: Tensor) -> Tensor:
        from function import Add
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(None, function=Add(self,other), shape=other.shape ,is_evaluated=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def __sub__(self, other:Tensor) -> Tensor:
        from function import Sub
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(None,function=Sub(self,other), shape=other.shape,is_evaluated=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def __mul__(self, other:Tensor) -> Tensor:
        from function import Mul
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(None,function=Mul(self,other), shape=other.shape,is_evaluated=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")

    def __div__(self, other:Tensor) -> Tensor:
        from function import Div
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(None,function=Div(self,other), shape=other.shape,is_evaluated=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")

    def __matmul__(self, other: Tensor) -> Tensor:
        from function import MatMul
        if isinstance(other, Tensor):
            # Ensure that matrix dimensions are compatible for multiplication
            if self.data.shape[-1] != other.data.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {self.data.shape} and {other.data.shape}")
            return Tensor(None,function=MatMul(self,other), shape=(self.data.shape[0], other.data.shape[1]), is_evaluated=False)
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def expand(self, size: int) -> Tensor:
        from function import Expand
        assert size >=1 , "size should be bigger than 1"
        return Tensor(None,function=Expand(self),shape=(size,*self.shape), is_evaluated=False)

    def maximum(self, gate: int) -> Tensor:
        from function import Maximum
        return Tensor(None, function=Maximum(self, gate), shape=self.shape, is_evaluated=False)

    def exp(self) -> Tensor:
        from function import Exp
        return Tensor(None, functon=Exp(self), shape=self.shape, is_evaluated=False)

    def sqrt(self) -> Tensor:
        from function import Sqrt
        return Tensor(None, function=Sqrt(self), shape=self.shape, is_evaluated=False)

    def max(self, dim:int) -> Tensor:
        from function import Max
        return Tensor(None, function=Max(self, dim), shape=self.shape[:dim] + self.shape[dim+1:], is_evaluated=False)
    
    def sum(self, dim:int) -> Tensor:
        from function import Sum
        return Tensor(None, function=Sum(self, dim), shape=self.shape[:dim] + self.shape[dim+1:], is_evaluated=False)
    
    def mseLoss(self) -> Tensor:
        from function import MSELoss
        return Tensor(None,function=MSELoss(self), shape=(1,),is_evaluated=False)

    def transpose(self) -> Tensor:
        pass

    def concate(self, dim:int) -> Tensor:
        pass
    
    def __repr__(self):
        return f"Tensor(data= {self.data}, grad= {self.gradient}, function: {self.function})"
    
    def backward(self, level:str = None):
        if self.function is None: return
        if self.gradient is None: 
            self.gradient = np.array(1)
            level = ""
        if self.is_evaluated:
            self.function.res = self.data
        assert self.gradient.shape == self.shape, "gradient's shape should equal to tensor shape"
        self.function.backward(self.gradient)

            

            