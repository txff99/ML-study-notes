import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestContiguous(unittest.TestCase):
    def test_contiguous_unchanged(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a.contiguous()
        self.assertTrue(a is b)

    def test_contiguous_unchanged_after_transpose(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a.transpose(2,1).transpose(2,1)
        self.assertTrue(b.is_contiguous()==True)
    
    @unittest.skip("not implemented")
    def test_contiguous_unchanged_after_slicing(self):
        pass

    def test_contiguous_after_transpose(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a.transpose(2,1)
        self.assertTrue(b.is_contiguous()==False)
        c = b.contiguous()
        self.assertTrue(c.is_contiguous()==True)
        self.assertTrue(c.is_realized==False)

    def test_contiguous_after_expand(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a.expand(3,2,3)
        self.assertTrue(b.is_contiguous()==False)
        c = b.contiguous()
        self.assertTrue(c is not b)
        self.assertTrue(c.is_contiguous()==True)
    
    def test_contiguous_numpy_impl(self):
        from runtime.numpy.contiguous import contiguous_numpy_impl
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw,shape=(3,3,1),strides=(3,1,3)) # make a fake tranposed tensor
        b = contiguous_numpy_impl(a)
        self.assertTrue(np.allclose(b, raw.transpose(0,2,1).flatten()))

if __name__ == "__main__":
    unittest.main()
