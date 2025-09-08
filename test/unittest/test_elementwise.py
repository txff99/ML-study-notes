import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestElementwise(unittest.TestCase):
    def test_exp(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        c = a.exp()
        self.assertTrue(c.shape==(3,1,3))
        self.assertTrue(np.allclose(c.numpy(),np.exp(raw)))

    def test_sqrt(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        c = a.sqrt()
        self.assertTrue(c.shape==(3,1,3))
        self.assertTrue(np.allclose(c.numpy(),np.sqrt(raw)))
    
    def test_noncontiguous_exp(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw).expand(3,2,3)
        c = a.exp()
        self.assertTrue(c.shape==(3,2,3))
        self.assertTrue(np.allclose(c.numpy(),np.exp(np.broadcast_to(raw,(3,2,3)))))
    
if __name__ == "__main__":
    unittest.main()
