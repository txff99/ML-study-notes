import numpy as np
from tensor import Tensor

LEVEL_BLANK = "|    "

"""
function use to store compute graph
"""

class Function:
    def __init__(self, *mem:Tensor):
        self.parents = list(mem)
        self.name: str = None
    
    def backward(self,grad:np.array):
        pass

    def __repr__(self):
        return self.name
    
    def printAST(self,level:str="   "):
        for mem in self.parents:
            if mem.function is None: print(level[:-3]+"-", "leaf", mem.shape)
            else:
                print(level[:-3]+"-", mem.function,mem.shape)
                mem.function.printAST(level=level+LEVEL_BLANK)


class Add(Function):
    def __init__(self,a:Tensor,b:Tensor):
        super().__init__(a,b)
        self.name = "add"

    def backward(self,grad:np.array):
        for mem in self.parents:
            mem.gradient = grad
            mem.backward()

class Sub(Function):
    def __init__(self,a:Tensor,b:Tensor):
        super().__init__(a,b)
        self.name = "sub"
    
    def backward(self,grad:np.array):
        for mem in self.parents:
            mem.gradient = grad
            mem.backward()

class MatMul(Function):
    def __init__(self,a:Tensor,b:Tensor):
        super().__init__(a,b)
        self.name = "matmul"
    
    def backward(self,grad:np.array):
        mem1, mem2 = self.parents
        mem1.gradient = grad @ mem2.numpy().T
        mem2.gradient = mem1.numpy().T @ grad
        mem1.backward()
        mem2.backward()

class MSELoss(Function):
    def __init__(self,a:Tensor):
        super().__init__(a)
        self.name = "mse"
    
    def backward(self,grad:np.array):
        assert grad.shape == (), "mse's gradient should be scalar"
        mem = self.parents[0]
        mem.gradient = grad * mem.numpy()*2/mem.numpy().size
        mem.backward()

class Expand(Function):
    def __init__(self, a:Tensor):
        super().__init__(a)
        self.name = "expd"
    
    def backward(self,grad:np.array):
        mem = self.parents[0]
        mem.gradient = np.sum(grad,axis=0)
        mem.backward()

class Max(Function):
    def __init__(self, a:Tensor, gate: int):
        super().__init__(a)
        self.name = "max"
        self.gate = gate
    
    def backward(self,grad:np.array):
        mem = self.parents[0]
        mask = mem.data > self.gate
        mem.gradient = grad * mask
        mem.backward()
    
