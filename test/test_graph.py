import numpy as np
import sys
import unittest

sys.path.append("..")
from tensor import Tensor
from graph import Graph
class TestGraph(unittest.TestCase):
    def test_graph_constructor(self):
        a = Tensor(1)
        b = Tensor(2)
        c = Tensor(3)
        d = a + b
        e = b + c
        f = d + e
        g = Graph(f)
        linearized: list[Tensor] = g.toposort()
        gt_linearized = [d,e,f]
        self.assertEqual(linearized,gt_linearized)
    
    def test_graph_constructor_visited(self):
        a = Tensor(1)
        b = Tensor(2)
        c = a + b
        d = a + c
        e = c + b
        f = d + e
        g = Graph(f)
        linearized: list[Tensor] = g.toposort()
        gt_linearized = [c, d, e, f] # c is visited so it only presents once
        self.assertEqual(linearized,gt_linearized)
    
    def test_graph_rewriter(self):
        def realize_tensor_pass(tensor: Tensor, args=None):
            tensor.is_realized = True
        a = Tensor(1)
        b = Tensor(2)
        c = a + b
        d = a + c
        g = Graph(d)
        g.rewrite(realize_tensor_pass)
        self.assertTrue(c.is_realized==True)
        self.assertTrue(d.is_realized==True)
    
    def test_graph_add_contiguous(self):
        print("Start test_graph_add_contiguous")
        from graph import add_contiguous_before_ari
        a = Tensor(None, shape=(3,2,3),strides=(6,0,3),is_realized=False)
        b = Tensor(None, shape=(3,2,3),strides=(6,0,3),is_realized=False)
        self.assertTrue(a.is_contiguous()==False)
        c = a + b
        g = Graph(c)
        g.rewrite(add_contiguous_before_ari)
        linearized = g.toposort()
        self.assertTrue(len(linearized)==3)
        self.assertTrue(linearized[1].is_contiguous()==True and linearized[2].is_contiguous()==True)

if __name__ == "__main__":
    unittest.main()