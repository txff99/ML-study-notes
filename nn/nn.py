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
    def __init__(self, in_feature:int, out_feature:int, bias:float|np.ndarray = None, weights=None):
        super().__init__()
        self.weights = weights if weights is not None else Tensor(np.random.rand(out_feature,in_feature))
        self.bias = None
        self.out_feature = out_feature
        self.in_feature = in_feature
        if bias is not None and isinstance(bias, float):
            self.bias = np.full((out_feature), bias)
        else:
            self.bias = Tensor(np.zeros((out_feature))) 
    
    def __call__(self,x:Tensor) -> Tensor:
        assert isinstance(x,Tensor), "input should be a Tensor"
        assert x.shape[-1] == self.weights.shape[1], f"tensor shape should be (...,{self.weights.shape[1]})"
        if len(x.shape) == 2:
            bias = self.bias.reshape(1, self.out_feature)
        else:
            bias = self.bias.reshape(1, 1, self.out_feature)
        return x @ self.weights.transpose(1,0) + bias
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return x.maximum(0)

class MLP(Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int):
        super().__init__()
        self.ll1 = Linear(in_dim, hidden_dim)
        self.ll2 = Linear(hidden_dim, out_dim)
        self.relu = ReLU()

    def __call__(self, x:Tensor) -> Tensor:
        return self.ll2(self.relu(self.ll1(x)))

class SoftMax(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x:Tensor, dim:int=-1) -> Tensor:
        # exp(xi - rowmax(xi)) / sum(exp(xi - rowmax(xi)))
        rowmax = x.max(dim=dim,keepdims=True)
        exp = (x-rowmax).exp()
        return exp / exp.sum(dim=dim,keepdims=True)
    
class SelfAttention(Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
    
    def __call__(self, x: Tensor, mask=None):
        """ 
            x : (b, seq, C)
        """
        q = self.q_proj(x)
        k_t = self.k_proj(x).transpose(2,1)
        v = self.v_proj(x)
        scale_factor = Tensor(1/np.sqrt(self.d_model)).expand(x.shape[0]).expand(x.shape[0])
        score = (q @ k_t) / scale_factor
        softmax = SoftMax()
        return softmax(score + mask) @ v

class MHA(Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        assert d_model % num_head == 0,"head dim should be divisable by d_model"
        self.num_head = num_head
        self.num_kv = num_kv
        self.d_model = d_model
        self.head_dim = d_model // num_head
        self.qkv_proj = Linear(d_model, 3*d_model)
        self.final_proj = Linear(d_model, d_model, True) 
        self.softmax = SoftMax()
    
    def __call__(self, x: Tensor, mask=None):
        """ 
            x : (b, seq, C)
        """
        qkv = self.qkv_proj(x) # (b, s, 3d)
        b, s, _ = qkv.shape
        qkv = qkv.reshape(b, s, 3, self.num_head, self.head_dim).transpose(3,1) # (b, h, 3, s, hd)
        q = qkv[:,:,0]
        k = qkv[:,:,1]
        v = qkv[:,:,2]
        scale_factor = Tensor(1/np.sqrt(self.d_model))
        score = q @ k.transpose(4,3)/scale_factor
        return self.final_proj(self.softmax(score + mask) @ v)




