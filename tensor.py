from __future__ import annotations
import numpy as np
from typing import List
from enum import Enum

class OpType(Enum):
    ADD = 1
    SUB = 2
    MATMUL = 3
    MSELOSS = 4 
    EXPAND = 5

FUNCTION_TO_OPTYPE = {
    "add": OpType.ADD,
    "sub": OpType.SUB,
    "matmul": OpType.MATMUL,
    "mse": OpType.MSELOSS,
    "expd": OpType.EXPAND
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
        self.data = np.asarray(data)
        self.dtype = dtype
        self.gradient: np.array = None
        self.function = function
        self.shape = self.data.shape if self.data is not None else shape
        self.is_evaluated = is_evaluated
        self.device = "cpu"
        self.device_ptr = None
    
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
        from runtime import GPU
        gpu = GPU()
        visited = set()
        def dfs(tensor: Tensor):
            if tensor in visited: return
            if tensor.device == "gpu": return
            tensor.device = "gpu"
            visited.add(tensor)
            gpu.alloc(tensor)
            if tensor.is_evaluated == True: 
                gpu.copyHTOD(tensor)
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
        dfs(self)        


    def toCPU(self, cleanup: bool = False):
        """copy gpu data back to cpu; clean up (optional)"""
        from runtime import GPU
        gpu = GPU()
        visited = set()
        def dfs(tensor: Tensor):
            if tensor in visited: return
            if tensor.device == "cpu": return
            tensor.device = "cpu"
            visited.add(tensor)    
            gpu.copyDTOH(tensor)
            if cleanup:
                gpu.free(tensor)
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
        dfs(self)        

    def evaluate(self) -> np.array:
        from runtime import CPU
        from runtime import GPU
        if self.device == "cpu":
            cpu = CPU(self.lower())
            cpu.run()
            return self.data
        elif self.device == "gpu":
            gpu = GPU(self.lower())
            gpu.run()
            self.toCPU(cleanup=True)
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
        return Tensor(None,function=Expand(self, Tensor(data=size)),shape=(size,*self.shape), is_evaluated=False)


    def mseLoss(self) -> Tensor:
        from function import MSELoss
        return Tensor(None,function=MSELoss(self), shape=(1,),is_evaluated=False)

    def __repr__(self):
        return f"Tensor(data= {self.data}, grad= {self.gradient}, function: {self.function})"
    
    def backward(self, level:str = None):
        if self.function is None: return
        if self.gradient is None: 
            self.gradient = np.array(1)
            level = ""
        assert self.gradient.shape == self.shape, "gradient's shape should equal to tensor shape"
        self.function.backward(self.gradient)

            

            