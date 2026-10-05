import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph, constant_folding, constant_realize, canonicalize
constant_folding_pipeline = [canonicalize, constant_folding, constant_realize]
class TestGraph(unittest.TestCase):
    def test_realize_constant(self):
        a = Tensor(1)
        b = Tensor(2) 
        c = Tensor(None, shape=(), is_realized=False)
        d = a + b
        e = c * d
        self.assertFalse(d.is_realized)
        self.assertFalse(e.is_realized)
        g = Graph(e)
        g.rewrite(constant_realize)
        self.assertTrue(d.is_realized)
        self.assertFalse(e.is_realized)

    def test_realize_constant(self):
        a = Tensor(1)
        b = Tensor(2) 
        c = Tensor(3)
        d = Tensor(None, shape=(), is_realized=False)
        h = a + b
        i = h + c
        j = h + d
        k = i + j
        graph = Graph(k)
        graph.rewrite(constant_realize)
        self.assertTrue(h.is_realized)
        self.assertTrue(i.is_realized)
        self.assertFalse(j.is_realized)
        self.assertFalse(k.is_realized)
    
    def test_add_sub1(self):
        a = Tensor(1)
        b = Tensor(2) 
        c = Tensor(None, shape=(), is_realized=False)
        d = a + c
        e = d + b
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[1].data, 3)

        d = a - c
        e = b + d
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[0].data, 3)
        self.assertEqual(e.function.name, "sub")

        d = a - c
        e = d + b
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[0].data, 3)
        self.assertEqual(e.function.name, "sub")

        d = a - c
        e = b - d
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[0].data, 1)
        self.assertEqual(e.function.name, "add")

        d = a - c
        e = d - b
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[0].data, -1)
        self.assertEqual(e.function.name, "sub")

    def test_mul_div(self):
        a = Tensor(1)
        b = Tensor(2) 
        c = Tensor(None, shape=(), is_realized=False)
        d = a * c
        e = d * b
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[1].data, 2)

        d = a / c
        e = b * d
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[0].data, 2)
        self.assertEqual(e.function.name, "div")

        d = a / c
        e = d * b
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[0].data, 2)
        self.assertEqual(e.function.name, "div")

        d = a / c
        e = b / d
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[0].data, 2)
        self.assertEqual(e.function.name, "mul")

        d = a / c
        e = d / b
        g = Graph(e)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(e.function.parents[0].data, 0.5)
        self.assertEqual(e.function.name, "div")
    
    def test_matmul1(self):
        np.random.seed(1)
        raw1 = np.random.rand(4,5)
        raw2 = np.random.rand(4,1)
        w = Tensor(raw1)
        v = Tensor(None, shape=(5,1), is_realized=False)
        c = Tensor(raw2)
        d = (w @ v) / c
        g = Graph(d)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(d.function.name, "matmul")
        self.assertTrue(d.function.parents[0].is_realized)
        self.assertTrue(np.allclose(d.function.parents[0].numpy(), raw1 / raw2))
        self.assertFalse(d.function.parents[1].is_realized)

    def test_matmul2(self):  
        np.random.seed(1)
        raw1 = np.random.rand(4,5)
        raw2 = np.random.rand(4,1)
        raw3 = np.random.rand(4,1)
        w = Tensor(raw1)
        v = Tensor(None, shape=(5,1), is_realized=False)
        b = Tensor(raw2) 
        c = Tensor(raw3)
        d = (w @ v + b) / c
        g = Graph(d)
        g.rewrite(constant_folding_pipeline)
        self.assertEqual(d.function.name, "add")
        self.assertTrue(d.function.parents[0].function.parents[0].is_realized)
        self.assertTrue(np.allclose(d.function.parents[0].function.parents[0].numpy(), raw1 / raw3))
        self.assertTrue(d.function.parents[1].is_realized)
        self.assertTrue(np.allclose(d.function.parents[1].numpy(), raw2/raw3))
    
    
if __name__ == "__main__":
    unittest.main()