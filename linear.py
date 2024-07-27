import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

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
    
def copy_weights(src: Linear, target: nn.Linear):
    target.weight = nn.Parameter(torch.from_numpy(src.weights.numpy().T).float())
    target.bias = nn.Parameter(torch.from_numpy(src.bias.numpy().T).float())
    
def test_output():
    input_tensor = np.random.rand(4,2)
    ll = Linear(2,3,bias=True)
    output_tensor = ll(Tensor(input_tensor))

    ll_gt = nn.Linear(2,3,bias=False)
    copy_weights(ll,ll_gt)
    output_tensor_gt = ll_gt(torch.from_numpy(input_tensor).float())
    
    print("result is:\n",output_tensor)
    print("gt is \n",output_tensor_gt)

def test_backward():
    input_tensor = np.random.rand(4,2)
    gt_tensor = Tensor(np.random.rand(4,3))
    
    ll = Linear(2,3,bias=True)
    output_tensor = ll(Tensor(input_tensor))
    l2 = (output_tensor-gt_tensor).l2()
    l2.backward()

if __name__ == "__main__":
    test_backward()

            