import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestExpand(unittest.TestCase):
    def test_singledim_success(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a.expand(3,2,3)
        diff = np.abs(b.numpy() - raw.repeat(2,axis=1)).max()
        self.assertTrue(np.allclose(b.numpy(),raw.repeat(2,axis=1)))
    
    def test_multidim_success(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3,1)
        a = Tensor(raw)
        b = a.expand(3,4,3,3)
        self.assertTrue(np.allclose(b.numpy(),raw.repeat(4,axis=1).repeat(3,axis=3)))
    
    @unittest.expectedFailure
    def test_invaliddim_fail(self):
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a.expand(3,2,6)


if __name__ == "__main__":
    unittest.main()
