import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor

class TestIndexPut(unittest.TestCase):
    def test_shape(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a[1:2,:,:]
        self.assertEqual(b.shape, (1,1,3))
        c = a[2,:,:]
        self.assertEqual(c.shape, (1,3))
    
    def test_strides(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a[1:2,:,:]
        self.assertEqual(b.strides, (3,3,1))
        c = a[2,:,:]
        self.assertEqual(c.strides, (3,1))
    
    def test_offset(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a[1:2,:,:]
        self.assertEqual(b.offset, 3)
        c = a[2,:,:]
        self.assertEqual(c.offset, 6)
    
    def test_contiguous(self):
        np.random.seed(1)
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a[1:2,:,:]
        self.assertTrue(b.is_contiguous()==False)
        self.assertTrue(np.allclose(b.numpy(), raw[1:2,:,:]))
        c = a[2,:,:]
        self.assertTrue(c.is_contiguous()==False)
        self.assertTrue(np.allclose(c.numpy(), raw[2,:,:]))

    def test_negdim(self):
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a[-1,:,:]
        self.assertTrue(np.allclose(b.numpy(), raw[-1,:,:]))
        c = a[-3:-1,:,:]
        self.assertTrue(np.allclose(c.numpy(), raw[-3:-1,:,:]))
        
    @unittest.expectedFailure
    def test_invaliddim_fail1(self):
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a[3,:,:]

    @unittest.expectedFailure
    def test_invaliddim_fail2(self):
        raw = np.random.rand(3,1,3)
        a = Tensor(raw)
        b = a[4:6,:,:]


if __name__ == "__main__":
    unittest.main()
