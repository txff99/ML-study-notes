import numpy as np
import sys
import unittest

sys.path.append("..")
from tensor import Tensor

class TestGPUAri(unittest.TestCase):
    def test_gpu_matmul(self):
        a = Tensor([[1,2],
                    [3,4]])
        b = Tensor([[2,2],
                    [1,3]])
        c = a @ b
        c.toGPU()
        np.testing.assert_array_equal(c.realize(), np.array([[4,8],
                                                [10,18]]))
    
    def test_gpu_add(self):
        a = Tensor([[1,2],
                    [3,4]])
        b = Tensor([[2,3],
                    [1,5]])
        c = a + b
        c.toGPU()
        np.testing.assert_array_equal(c.realize(), np.array([[3,5],
                                                [4,9]]))
    
    def test_gpu_sub(self):
        a = Tensor([[1,2],
                    [3,4]])
        b = Tensor([[2,3],
                    [1,5]])
        c = a - b
        c.toGPU()
        np.testing.assert_array_equal(c.realize(), np.array([[-1,-1],
                                                [2,-1]]))
    def test_gpu_expand(self):
        a = Tensor([1,2])
        c = a.expand(2)
        c.toGPU()
        np.testing.assert_array_equal(c.realize(), np.array([[1,2],
                                                [1,2]]))

if __name__ == "__main__":
    unittest.main()