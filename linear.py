import numpy as np
from Tensor import Tensor

class Linear:
    def __init__(self, in_feature:int, out_feature:int, bias:bool = False):
        self.weights = Tensor(np.random.uniform(-np.sqrt(6/in_feature),np.sqrt(6/in_feature),(in_feature,out_feature)))
        self.bias = Tensor(np.zeros(out_feature))
        if bias == True:
            self.bias = Tensor(np.random.rand(out_feature))
    
    def __call__(self,x:Tensor) -> Tensor:
        assert isinstance(x,Tensor), "input should be a Tensor"
        assert x.shape[-1] == self.weights.shape[0], f"tensor shape should be (...,{self.weights.shape[0]})"
        return x @ self.weights + self.bias
    

            