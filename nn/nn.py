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
        return x.maximum(0)

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
        rowmax = x.max(dim=1)
        exp = (x-rowmax).exp()
        return exp / exp.sum(dim=1)
    
class SelfAttention(Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = Linear(d_model, d_model, True)
        self.k_proj = Linear(d_model, d_model, True)
        self.v_proj = Linear(d_model, d_model, True)
    
    def __call__(self, x: Tensor):
        """ 
            x : (seq, C)
        """
        q = self.q_proj(x)
        k_t = self.k_proj(x).transpose()
        v = self.v_proj(x)
        scale_factor = Tensor(1/np.sqrt(self.d_model)).expand(x.shape[0]).expand(x.shape[0])
        score = (q @ k_t) / scale_factor
        softmax = SoftMax()
        return softmax(score) @ v

class MHA(Module):
    def __init__(self, d_model, num_head, num_kv):
        super().__init__()
        assert d_model % num_head == 0 and d_model % num_kv == 0,"head dim and kv dim should be divisable by d_model"
        assert num_kv <= num_head, "kv_dim should be smaller than head_dim"
        assert num_head % num_kv == 0, "head_dim should be divisable by kv_dim"
        
        self.num_head = num_head
        self.num_kv = num_kv
        self.d_model = d_model
        self.head_dim = d_model // num_head
        self.kv_rep = num_head // num_kv
        self.q_proj = [Linear(d_model, self.head_dim, True) for _ in range(self.num_head)]
        self.k_proj = [Linear(d_model, self.head_dim, True) for _ in range(self.num_kv)]
        self.v_proj = [Linear(d_model, self.head_dim, True) for _ in range(self.num_kv)]
        self.final_proj = Linear(d_model, d_model, True) 
    
    def __call__(self, x: Tensor):
        """ 
            x : (seq, C)
        """
        res = Tensor(None, shape=x.shape)
        for i in range(self.num_head):
            q = self.q_proj[i](x)
            k_t = self.k_proj[i // self.kv_rep](x).transpose()
            v = self.v_proj[i // self.kv_rep](x)
            scale_factor = Tensor(1/np.sqrt(self.d_model)).expand(x.shape[0]).expand(x.shape[0])
            score = (q @ k_t) / scale_factor
            softmax = SoftMax()
            res.concate(softmax(score) @ v, dim = -1)
        return self.final_proj(res)




