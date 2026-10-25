import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import sint, Tensor
class TestShape(unittest.TestCase):
    def test_sint_int(self):
        s1 = sint(max=6,min=1)
        s2 = s1 * 2
        s3 = s2 + 2
        self.assertTrue(s3.min==4 and s3.max==14 and s3.function.parents[0]==s2 and s3.function.parents[0].function.parents[0]==s1)

    def test_sint_sint(self):
        s1 = sint(max=6,min=1)
        s2 = sint(max=6,min=1)
        s3 = s2 + s1
        
    def test_expand(self):
        b = sint(max=6,min=1)
        t1 = Tensor(None, shape=(b, 3, 1), is_realized=False)
        t1_T = t1.expand(b,3,3)
        self.assertEqual(t1_T.strides,(3,1,0))
        t2 = Tensor(None, shape=(3, b, 3, 1), is_realized=False)
        t2_T = t2.expand(3, b,3,3)
        # self.assertEqual(t2_T.strides,(sint(scale=3, bias=0, min=3, max=18,val=b),3,1,0))

    def test_is_contiguous(self):
        b = sint(max=6,min=1)
        t1 = Tensor(None, shape=(b, 3, 1), is_realized=False)
        self.assertTrue(t1.is_contiguous())
        t1_T = t1.expand(b,3,3)
        # after expand tensor is not contiguous
        self.assertFalse(t1_T.is_contiguous())
    
    def test_contiguous(self):
        # contiguous always running on the runtime so it will always get the real shape
        b = sint(max=6,min=1)
        t1 = Tensor(None, shape=(b, 3, 1), is_realized=False)
        self.assertTrue(t1.is_contiguous())
        t1_T = t1.expand(b,3,3)
        t1_c = t1_T.contiguous()
    
    def test_shape_realize(self):
        b = sint(max=6,min=1)
        t1 = Tensor(None, shape=(b, 3, 1), is_realized=False)
        b.bind(3)
        self.assertEqual(t1.shape, (3,3,1))

    def test_realize(self):
        b = sint(max=6,min=1)
        t1 = Tensor(None, shape=(b, 3, 1), is_realized=False)
        t2 = Tensor(np.random.rand(1,3,1))
        t3 = t1 + t2
        real = Tensor(np.random.rand(3,3,1))
        b.bind(3)
        t3.realize()
        self.assertTrue(t3.shape==(3,3,1) and t3.strides==(3,1,1))
    
    def test_export_program(self):
        from graph import Graph
        np.random.seed(1)
        src = np.random.rand(32,4)
        raw1 = np.random.rand(4, 32) 
        raw2 = np.random.rand(32, 32)  
        inp = Tensor(src)
        b = sint(max=32,min=1)
        x = Tensor(None, shape=(b,4), is_realized=False)
        weight = Tensor(raw1)
        bias = Tensor(raw2)
        y = x @ weight + bias
        g = Graph(y)
        program = g.export(inputs=(x,)) # List[Op]
        out = program.run((inp,))

        self.assertTrue(np.allclose(out[0].numpy(), src @ raw1 + raw2))

        # replay 
        np.random.seed(2)
        src2 = np.random.rand(32,4)
        inp2 = Tensor(src2)
        out = program.run((inp2,))
        self.assertTrue(np.allclose(out[0].numpy(), src2 @ raw1 + raw2))


if __name__ == "__main__":
    unittest.main()