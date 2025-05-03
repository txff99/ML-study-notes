
import numpy as np
import torch
import torch.nn as nn
import sys
import unittest
sys.path.append("..")
from tensor import Tensor
from nn.nn import Linear, ReLU

np.set_printoptions(precision=4, suppress=True)

def copy_weights(src: Linear, target: nn.Linear):
    target.weight = nn.Parameter(torch.from_numpy(src.weights.numpy().T).float())
    target.bias = nn.Parameter(torch.from_numpy(src.bias.numpy().T).float())

class TestLinear(unittest.TestCase): 
    def test_linear_cpu(self):
        print("==========Testing output=========")
        input_tensor = np.random.rand(4,2)
        ll = Linear(2,3,bias=True)
        output_tensor = ll(Tensor(input_tensor)).evaluate()

        ll_gt = nn.Linear(2,3,bias=False)
        copy_weights(ll,ll_gt)
        output_tensor_gt = ll_gt(torch.from_numpy(input_tensor).float()).detach().numpy()
        
        np.testing.assert_allclose(output_tensor, output_tensor_gt, rtol=1e-5, atol=1e-8)

    
    def test_linear_gpu(self):
        print("==========Testing output=========")
        input_tensor = np.random.rand(4,2)
        ll = Linear(2,3,bias=True)
        
        output_tensor = ll(Tensor(input_tensor))
        output_tensor.toGPU()
        output_tensor.evaluate()

        ll_gt = nn.Linear(2,3,bias=False)
        copy_weights(ll,ll_gt)
        output_tensor_gt = ll_gt(torch.from_numpy(input_tensor).float()).detach().numpy()
        np.testing.assert_allclose(output_tensor.numpy(), output_tensor_gt, rtol=1e-5, atol=1e-8)
    

class TestReLU(unittest.TestCase): 
    def test_relu_cpu(self):
        print("==========Testing output=========")
        input_tensor = np.random.rand(4,2)
        relu = ReLU()
        output_tensor = relu(Tensor(input_tensor)).evaluate()

        relu_gt = nn.ReLU()
        output_tensor_gt = relu_gt(torch.from_numpy(input_tensor).float()).detach().numpy()
        
        np.testing.assert_allclose(output_tensor, output_tensor_gt, rtol=1e-5, atol=1e-8)

    
    def test_linear_gpu(self):
        print("==========Testing output=========")
        input_tensor = np.random.rand(4,2)
        relu = ReLU()
        output_tensor = relu(Tensor(input_tensor)).
        output_tensor.toGPU()
        output_tensor.evaluate()

        relu_gt = nn.ReLU()
        output_tensor_gt = relu_gt(torch.from_numpy(input_tensor).float()).detach().numpy()
        
        np.testing.assert_allclose(output_tensor.numpy(), output_tensor_gt, rtol=1e-5, atol=1e-8)

def test_backward():
    print("==========Testing backward=========")
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
    
    print("------ASTtree----")
    l2.function.printAST()
    print("==========result===========")

    if np.allclose(ll.weights.gradient, ll_gt.weight.grad.numpy().T) and \
        np.allclose(ll.bias.gradient, ll_gt.bias.grad.numpy()):   print("PASSED")
    else: print("FAILED")

if __name__ == "__main__":
    unittest.main()