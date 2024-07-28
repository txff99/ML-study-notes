from __future__ import annotations
import numpy as np


class Tensor:
    def __init__(self,data:np.array,function=None):
        self.data: np.array = data
        self.gradient: np.array = None
        self.function = function
        self.shape = self.data.shape
    
    def numpy(self) -> np.array:
        return self.data
    
    def transpose(self):
        self.data = self.data.T
        self.shape = self.data.shape

    def __add__(self, other: Tensor) -> Tensor:
        from function import Add
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(self.data + other.numpy(),function=Add(self,other))
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def __sub__(self, other:Tensor) -> Tensor:
        from function import Sub
        assert self.shape==other.shape , "tensor shape is not the same"
        if isinstance(other, Tensor):
            return Tensor(self.data - other.numpy(),function=Sub(self,other))
        else:
            raise TypeError("The operand must be an instance of Tensor")

    def __matmul__(self, other: Tensor) -> Tensor:
        from function import MatMul
        if isinstance(other, Tensor):
            # Ensure that matrix dimensions are compatible for multiplication
            if self.data.shape[-1] != other.data.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {self.data.shape} and {other.data.shape}")
            result = self.data @ other.data
            return Tensor(result,function=MatMul(self,other))
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def expand(self, size: int) -> Tensor:
        from function import Expand
        assert size >=1 , "size should be bigger than 1"
        return Tensor(np.repeat(np.expand_dims(self.data,axis=0),size,axis=0),function=Expand(self))


    def mseLoss(self) -> Tensor:
        from function import MSELoss
        return Tensor(np.mean(self.data**2),function=MSELoss(self))

    def __repr__(self):
        return f"Tensor(data=\n {self.data}, grad=\n {self.gradient}, function: {self.function})"
    
    def backward(self, level:str = None):
        if self.function is None: return
        if self.gradient is None: 
            self.gradient = np.array(1)
            level = ""
        assert self.gradient.shape == self.shape, "gradient's shape should equal to tensor shape"
        self.function.backward(self.gradient)

            

            