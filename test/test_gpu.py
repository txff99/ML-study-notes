import numpy as np
import sys
import unittest

sys.path.append("..")
from tensor import Tensor

class TestCPUAri(unittest.TestCase):
    
    def test_gpu_matmul(self):
        a = Tensor([[1,2],
                    [3,4]])
        b = Tensor([[2],
                    [1]])
        c = a @ b
        np.testing.assert_array_equal(c.evaluate(), np.array([[4],
                                                [10]]))
    

if __name__ == "__main__":
    unittest.main()