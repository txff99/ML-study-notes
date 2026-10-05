import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestWhere(unittest.TestCase):
    def test_where(self):
        np.random.seed(1)
        raw1 = np.random.rand(3,3)
        raw2 = np.random.rand(3,3)
        raw3 = np.triu(np.ones((3,3)), k=0)
        mask = Tensor(raw3)
        a = Tensor(raw1)
        b = Tensor(raw2)
        c = a.where(mask, b)
        self.assertTrue(np.allclose(c.numpy(),np.where(raw3, raw1, raw2)))

if __name__ == "__main__":
    unittest.main()
