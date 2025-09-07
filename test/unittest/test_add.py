import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestAdd(unittest.TestCase):
    def test_simple_add_success(self):
        np.random.seed(1)
        raw1 = np.random.rand(3,1,3)
        np.random.seed(2)
        raw2 = np.random.rand(3,1,3)
        a = Tensor(raw1)
        b = Tensor(raw2)
        c = a+b
        self.assertTrue(np.allclose(c.numpy(),raw1+raw2))
    
    def test_noncontiguous_add(self):
        np.random.seed(1)
        raw1 = np.random.rand(3,1,3)
        np.random.seed(2)
        raw2 = np.random.rand(3,1,1)
        a = Tensor(raw1)
        b = Tensor(raw2).expand(3,1,3)
        c = a+b
        self.assertTrue(np.allclose(c.numpy(),raw1+raw2.broadcast_to(3,1,3)))
    
    @unittest.expectedFailure
    def test_invaliddim_fail(self):
        np.random.seed(1)
        raw1 = np.random.rand(3,1,3)
        np.random.seed(2)
        raw2 = np.random.rand(3,1,2)
        a = Tensor(raw1)
        b = Tensor(raw2)
        c = a+b

if __name__ == "__main__":
    unittest.main()
