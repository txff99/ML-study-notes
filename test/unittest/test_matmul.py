import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestMMA(unittest.TestCase):
    def test_simple_mma(self):
        np.random.seed(1)
        raw1 = np.random.rand(3,4)
        np.random.seed(2)
        raw2 = np.random.rand(4,3)
        a = Tensor(raw1)
        b = Tensor(raw2)
        c = a @ b
        self.assertTrue(np.allclose(c.numpy(), raw1 @ raw2))
    
    def test_noncontiguous_mma(self):
        np.random.seed(1)
        raw1 = np.random.rand(3,4)
        np.random.seed(2)
        raw2 = np.random.rand(1,3)
        a = Tensor(raw1)
        b = Tensor(raw2).expand(4,3)
        c = a @ b
        self.assertTrue(np.allclose(c.numpy(),raw1 @ np.broadcast_to(raw2,(4,3))))
    
    @unittest.expectedFailure
    def test_invaliddim_fail(self):
        np.random.seed(1)
        raw1 = np.random.rand(3,4)
        np.random.seed(2)
        raw2 = np.random.rand(3,3)
        a = Tensor(raw1)
        b = Tensor(raw2)
        c = a @ b

if __name__ == "__main__":
    unittest.main()
