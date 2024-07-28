
import numpy as np
import torch
import torch.nn as nn
from linear import Linear
from Tensor import Tensor

np.set_printoptions(precision=4, suppress=True)

def copy_weights(src: Linear, target: nn.Linear):
    target.weight = nn.Parameter(torch.from_numpy(src.weights.numpy().T).float())
    target.bias = nn.Parameter(torch.from_numpy(src.bias.numpy().T).float())
    
def test_output():
    print("---------Testing output--------")
    input_tensor = np.random.rand(4,2)
    ll = Linear(2,3,bias=True)
    output_tensor = ll(Tensor(input_tensor)).numpy()

    ll_gt = nn.Linear(2,3,bias=False)
    copy_weights(ll,ll_gt)
    output_tensor_gt = ll_gt(torch.from_numpy(input_tensor).float()).detach().numpy()
    
    print("is:\n",output_tensor)
    print("gt is \n",output_tensor_gt)

    print("---------result----------")
    if np.allclose(output_tensor, output_tensor_gt):   print("PASSED")
    else: print("FAILED")
    

def test_backward():
    print("---------Testing backward-------")
    input_tensor = np.random.rand(4,2)
    gt_tensor = np.random.rand(4,3)
    
    ll = Linear(2,3,bias=True)
    output_tensor = ll(Tensor(input_tensor))
    l2 = (output_tensor-Tensor(gt_tensor)).mseLoss()
    l2.backward()

    ll_gt = nn.Linear(2,3,bias=False)
    copy_weights(ll,ll_gt)
    output_tensor_gt = ll_gt(torch.from_numpy(input_tensor).float())
    output_tensor_gt.retain_grad()
    l2_gt = nn.MSELoss()(torch.from_numpy(gt_tensor).float(),output_tensor_gt)
    l2_gt.backward()

    print("mse:", l2.numpy())
    print("mse gt:", l2_gt.detach().numpy())
    print(f"gradient :\n weights {ll.weights.gradient}\n bias {ll.bias.gradient}")
    print(f"gt gradient :\n weights {ll_gt.weight.grad}\n bias {ll_gt.bias.grad}")
    
    print("---------result----------")

    if np.allclose(ll.weights.gradient, ll_gt.weight.grad.numpy().T) and \
        np.allclose(ll.bias.gradient, ll_gt.bias.grad.numpy()):   print("PASSED")
    else: print("FAILED")

if __name__ == "__main__":
    test_output()
    test_backward()