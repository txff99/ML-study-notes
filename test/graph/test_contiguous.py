import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph, add_contiguous_before_ari
class TestGraph(unittest.TestCase):
    def test_graph_add_contiguous(self):
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