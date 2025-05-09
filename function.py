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
        self.res: np.array = None
    
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

class Mul(Function):
    def __init__(self,a:Tensor,b:Tensor):
        super().__init__(a,b)
        self.name = "mul"
    
    def backward(self,grad:np.array):
        mem1, mem2 = self.parents
        mem1.gradient = grad * mem2.numpy()
        mem2.gradient = mem1.numpy() * grad
        mem1.backward()
        mem2.backward()

class Div(Function):
    def __init__(self,a:Tensor,b:Tensor):
        super().__init__(a,b)
        self.name = "div"
    
    def backward(self,grad:np.array):
        """ 
            c = b/a 
            dc / db = 1/a
            dc / da = -b/a**2
        """
        divdend, divsor = self.parents
        divdend.gradient = 1/divsor.numpy()
        divsor.gradient = -divdend/divsor.numpy()**2
        divdend.backward()
        divsor.backward()

class Sqrt(Function):
    def __init__(self, a: Tensor):
        super().__init__(a)
        self.name = "sqrt"
    
    def backward(self, grad:np.array):
        # a = sqrt(b)
        # da/db = -1/2*sqrt(b)
        mem = self.parents[0]
        mem.grad = -1/2*self.res

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
        self.name = "expand"
    
    def backward(self,grad:np.array):
        mem = self.parents[0]
        mem.gradient = np.sum(grad,axis=0)
        mem.backward()

class Maximum(Function):
    def __init__(self, a:Tensor, gate: int):
        super().__init__(a)
        self.name = "maximum"
        self.gate = gate
    
    def backward(self,grad:np.array):
        mem = self.parents[0]
        mask = mem.numpy() > self.gate
        mem.gradient = grad * mask
        mem.backward()

class Exp(Function):
    def __init__(self, a:Tensor):
        super().__init__(a)
        self.name = "exp"
    
    def backward(self,grad:np.array):
        mem = self.parents[0]
        mem.gradient = grad * self.res
        mem.backward()

class Max(Function):
    def __init__(self, a:Tensor, dim: int):
        # (N, dim) -> (N,)
        super().__init__(a)
        self.name = "max"
        self.dim = dim

    def backward(self, grad: np.array):
        # only works for dim=1
        mem = self.parents[0]
        mask = [mem[row]==max_val for row,max_val in enumerate(self.res)]
        mem.gradient = grad * mask
        mem.backward()

class Sum(Function):
    def __init__(self, a:Tensor, dim: int):
        # (N, dim) -> (N,)
        super().__init__(a)
        self.name = "sum"
        self.dim = dim

    def backward(self, grad: np.array):
        # only works for dim=1
        mem = self.parents[0]
        mem.gradient = grad
        mem.backward()
