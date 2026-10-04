import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph, constant_folding, constant_realize
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
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[1].data, 3)

        d = a - c
        e = b + d
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[0].data, 3)
        self.assertEqual(e.function.name, "sub")

        d = a - c
        e = d + b
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[0].data, 3)
        self.assertEqual(e.function.name, "sub")

        d = a - c
        e = b - d
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[0].data, 1)
        self.assertEqual(e.function.name, "add")

        d = a - c
        e = d - b
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[0].data, -1)
        self.assertEqual(e.function.name, "sub")

    def test_mul_div(self):
        a = Tensor(1)
        b = Tensor(2) 
        c = Tensor(None, shape=(), is_realized=False)
        d = a * c
        e = d * b
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[1].data, 2)

        d = a / c
        e = b * d
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[0].data, 2)
        self.assertEqual(e.function.name, "div")

        d = a / c
        e = d * b
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[0].data, 2)
        self.assertEqual(e.function.name, "div")

        d = a / c
        e = b / d
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[0].data, 2)
        self.assertEqual(e.function.name, "mul")

        d = a / c
        e = d / b
        g = Graph(e)
        g.rewrite(constant_folding)
        g.rewrite(constant_realize)
        self.assertEqual(e.function.parents[0].data, 0.5)
        self.assertEqual(e.function.name, "div")
    
    
if __name__ == "__main__":
    unittest.main()