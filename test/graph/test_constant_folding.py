import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph, constant_folding
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
        g.rewrite(constant_folding)
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
        graph.rewrite(constant_folding)
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
        g.rewrite(constant_folding)
        self.assertEqual(e.function.parents[1].data, 3)
    
    def test_add_sub2(self):
        a = Tensor(1)
        b = Tensor(2) 
        c = Tensor(None, shape=(), is_realized=False)
        d = a - c
        e = b + d
        g = Graph(e)
        g.rewrite(constant_folding)
        self.assertEqual(e.function.parents[0].data, 3)
        self.assertEqual(e.function.name, "sub")
   
    def test_add_sub3(self):
        a = Tensor(1)
        b = Tensor(2) 
        c = Tensor(None, shape=(), is_realized=False)
        d = a - c
        e = d + b
        g = Graph(e)
        g.rewrite(constant_folding)
        self.assertEqual(e.function.parents[0].data, 3)
        self.assertEqual(e.function.name, "sub")

if __name__ == "__main__":
    unittest.main()