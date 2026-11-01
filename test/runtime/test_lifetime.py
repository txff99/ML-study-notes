import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph
class TestRuntime(unittest.TestCase):
    def test_lifetime(self):
        x = Tensor(None, shape=(32,4), is_realized=False)
        weight = Tensor(None, shape=(4,32), is_realized=False)
        bias = Tensor(None, shape=(32,32), is_realized=False)
        y = x @ weight 
        z = y + bias
        g = Graph(z)
        program = g.export(inputs=(x,))
        snapshots = program.life_snapshots
        """
        add 94d0 (32, 32)
            - dfd0 matmul (32, 32) [nan]
            | - c650 (32, 4) [nan]
            | - 1c90 (4, 32) [nan]
            - 2c10 (32, 32) [nan]
        (alloc: dfd0, dealloc: []) (alloc: 94d0, dealloc: ['dfd0']) 
        """
        alloc1 = snapshots[0][0]
        dealloc1 = snapshots[0][1]
        self.assertTrue(y==(alloc1[0]))
        self.assertTrue(len(dealloc1)==0)
        alloc2 = snapshots[1][0]
        dealloc2 = snapshots[1][1]
        self.assertTrue(z==(alloc2[0]))
        self.assertTrue(y==(dealloc2[0]))

if __name__ == "__main__":
    unittest.main()