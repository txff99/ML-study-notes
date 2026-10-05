import numpy as np
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.append("../..")
from tensor import Tensor
from nn.nn import Linear, MLP, ReLU, SoftMax, SelfAttention, MHA
class TestNN(unittest.TestCase):
    def test_linear(self):
        np.random.seed(1)
        raw1 = np.random.rand(2, 4, 32)  # (b,s,d)

        x1 = torch.tensor(raw1, dtype=torch.float32)
        x2 = Tensor(raw1)

        linear = Linear(32, 64)
        linear_gt = torch.nn.Linear(32, 64)

        with torch.no_grad():
            linear_gt.weight.copy_(torch.tensor(linear.weights.numpy()))
            linear_gt.bias.copy_(torch.tensor(linear.bias.numpy()))

        out = linear(x2)
        out_gt = linear_gt(x1)
        self.assertTrue(np.allclose(out.numpy(), out_gt.detach().numpy()))
    
    def test_softmax(self):
        np.random.seed(1)
        raw1 = np.random.rand(2, 4, 32)  # (b,s,d)

        x1 = torch.tensor(raw1, dtype=torch.float32)
        x2 = Tensor(raw1)

        softmax = SoftMax()

        out = softmax(x2, dim=-1)
        out_gt = F.softmax(x1, dim=-1)
        self.assertTrue(np.allclose(out.numpy(), out_gt.detach().numpy()))

    def test_relu(self):
        np.random.seed(1)
        raw1 = np.random.rand(2, 4, 32)  # (b,s,d)

        x1 = torch.tensor(raw1, dtype=torch.float32)
        x2 = Tensor(raw1)

        relu = ReLU()
        relu_gt = torch.nn.ReLU()

        out = relu(x2)
        out_gt = relu_gt(x1)
        self.assertTrue(np.allclose(out.numpy(), out_gt.detach().numpy()))




if __name__ == "__main__":
    unittest.main()