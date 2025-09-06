import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestTranspose(unittest.TestCase):
    def test_strides_shape_data(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a.transpose(2,1)
        self.assertTrue(b.shape==(3,3,1))
        self.assertTrue(b.strides==(3,1,3))
        b.realize()
        self.assertTrue(np.allclose(a.data,b.data))
    
    def test_transpose_twice(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3,1)
        a = Tensor(raw)
        b = a.transpose(2,1).transpose(2,1)
        self.assertTrue(b.shape==(3,1,3,1))
        self.assertTrue(b.strides==(3,3,1,1))
        b.realize()
        self.assertTrue(np.allclose(a.data,b.data))
    
    @unittest.expectedFailure
    def test_invaliddim_fail(self):
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a.transpose(3,1)


if __name__ == "__main__":
    unittest.main()
