import numpy as np
import sys
sys.path.append("..")
from tensor import Tensor

class Module:
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Each module must implement its own forward pass.")

class Linear(Module):
    def __init__(self, in_feature:int, out_feature:int, bias:bool = False):
        super().__init__()
        self.weights = Tensor(np.random.uniform(-np.sqrt(6/in_feature),np.sqrt(6/in_feature),(in_feature,out_feature)))
        self.bias = Tensor(np.zeros(out_feature))
        if bias == True:
            self.bias = Tensor(np.random.rand(out_feature))
    
    def __call__(self,x:Tensor) -> Tensor:
        assert isinstance(x,Tensor), "input should be a Tensor"
        assert x.shape[-1] == self.weights.shape[0], f"tensor shape should be (...,{self.weights.shape[0]})"
        return x @ self.weights + self.bias.expand(x.shape[0])
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return x.max(0)

class MLP(Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int):
        super().__init__()
        self.ll1 = Linear(in_dim, hidden_dim, True)
        self.ll2 = Linear(hidden_dim, out_dim, True)
        self.relu = ReLU()

    def __call__(self, x:Tensor) -> Tensor:
        return self.ll2(self.relu(self.ll1(x)))

class SoftMax(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x:Tensor) -> Tensor:
        # exp(xi - rowmax(xi)) / sum(exp(xi - rowmax(xi)))
        
        return 
