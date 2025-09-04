import numpy as np
import sys
import unittest

sys.path.append("..")
from tensor import Tensor

class TestCPUAri(unittest.TestCase):
    def test_cpu_alu1(self):
        a = Tensor(1)
        b = Tensor(2)
        c = a + b
        c.realize()
        d = a - b
        d.realize()
        self.assertEqual(c.data, 3)
        self.assertEqual(d.data, -1)
    
    def test_cpu_matmul(self):
        a = Tensor([[1,2],
                    [3,4]])
        b = Tensor([[2],
                    [1]])
        c = a @ b
        np.testing.assert_array_equal(c.realize(), np.array([[4],
                                                [10]]))
    
    def test_cpu_expand(self):
        a = Tensor([1])
        b = a.expand(2)
        np.testing.assert_array_equal(b.realize(), np.array([[1],[1]]))

    def test_cpu_mse(self):
        a = Tensor([2,2])
        b = Tensor([4,4])
        l2 = (a - b).mseLoss()
        self.assertEqual(l2.realize(), 4)


if __name__ == "__main__":
    unittest.main()