import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestIndexPut(unittest.TestCase):
    def test_correctness(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        raw_test = raw.copy()
        a = Tensor(raw)
        raw_row = np.random.rand(1,1,3)
        new_row = Tensor(raw_row)
        a[1:2,:,:] = new_row
        raw_test[1:2,:,:] = raw_row
        self.assertTrue(np.allclose(a.numpy(), raw_test))


if __name__ == "__main__":
    unittest.main()
