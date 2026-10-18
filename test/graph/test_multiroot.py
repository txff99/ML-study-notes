import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph
class TestGraph(unittest.TestCase):
    def test_multiroots(self):
        a = Tensor(None, shape=(3,2,3),strides=(6,0,3),is_realized=False)
        b = Tensor(None, shape=(3,2,3),strides=(6,0,3),is_realized=False)
        self.assertTrue(a.is_contiguous()==False)
        c = a + b
        d = a * b
        g = Graph(c,d)
        linearized = g.toposort()
        self.assertTrue(len(linearized)==2)

if __name__ == "__main__":
    unittest.main()