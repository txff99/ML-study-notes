from __future__ import annotations
import numpy as np

LEVEL_BLANK = "   "
DEBUG=True

class Tensor:
    def __init__(self,data:np.array,tracer:dict=None):
        self.data: np.array = data
        self.gradient: np.array = None
        self.Tracer = tracer
        self.shape = self.data.shape
    
    def numpy(self) -> np.array:
        return self.data
    
    def transpose(self):
        self.data = self.data.T
        self.shape = self.data.shape

    def __add__(self, other: Tensor) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self.data + other.numpy(),tracer={"add":(self,other)})
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def __sub__(self, other:Tensor) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(self.data - other.numpy(),tracer={"sub":(self,other)})
        else:
            raise TypeError("The operand must be an instance of Tensor")

    def __matmul__(self, other: Tensor) -> Tensor:
        if isinstance(other, Tensor):
            # Ensure that matrix dimensions are compatible for multiplication
            if self.data.shape[-1] != other.data.shape[0]:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {self.data.shape} and {other.data.shape}")
            result = self.data @ other.data
            return Tensor(result,tracer={"matmul":(self,other)})
        else:
            raise TypeError("The operand must be an instance of Tensor")
    
    def mseLoss(self) -> Tensor:
        return Tensor(np.mean(self.data**2),tracer={"mse":(self,)})

    def __repr__(self):
        return f"Tensor(data=\n {self.data}, grad=\n {self.gradient}, Tracer: {self.Tracer.keys() if self.Tracer is not None else None})"
    
    def backward(self, gradient:np.array = None, level:str = None):
        if gradient is None: 
            gradient = np.array(1)
            level = ""
        if gradient.shape != self.shape:
            # to do: need a more general form 
            gradient = np.sum(gradient,axis=0)
        self.gradient = gradient
        if self.Tracer is None: 
            if DEBUG: print(level,"leaf",gradient.shape)
            return
        for op,children in self.Tracer.items():
            if DEBUG:
                print(level, op, gradient.shape)
            if op == "add" or op == "sub":
                children[0].backward(self.gradient,level=level+LEVEL_BLANK)
                children[1].backward(self.gradient,level=level+LEVEL_BLANK)
            elif op == "matmul":
                children[0].backward(self.gradient @ children[1].numpy().T,level=level+LEVEL_BLANK)
                children[1].backward(children[0].numpy().T @ self.gradient,level=level+LEVEL_BLANK)
            elif op == "mse":
                # to do: add prev gradient
                assert gradient.shape == (), "mse's gradient should be scalar"
                children[0].backward(gradient * children[0].numpy()*2/(children[0].numpy().size),level=level+LEVEL_BLANK)

    
        

            

            