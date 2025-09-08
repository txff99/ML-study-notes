import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestReduce(unittest.TestCase):
    def test_sum(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        c = a.sum(dim=2)
        self.assertTrue(c.shape==(3,1))
        self.assertTrue(np.allclose(c.numpy(),raw.sum(axis=2)))

    def test_max(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        c = a.max(dim=2)
        self.assertTrue(c.shape==(3,1))
        self.assertTrue(np.allclose(c.numpy(),raw.max(axis=2)))
    
    def test_noncontiguous_sum(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw).expand(3,2,3)
        c = a.sum(dim=1)
        self.assertTrue(c.shape==(3,3))
        self.assertTrue(np.allclose(c.numpy(),np.broadcast_to(raw,(3,2,3)).sum(axis=1)))
    
    @unittest.expectedFailure
    def test_invaliddim_fail(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        c = a.max(dim=3)

if __name__ == "__main__":
    unittest.main()
