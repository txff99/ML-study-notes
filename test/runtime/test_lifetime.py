import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph
class TestRuntime(unittest.TestCase):
    def test_lifetime(self):
        def get_id(x):
            return hex(id(x))[-4:]
        x = Tensor(None, shape=(32,4), is_realized=False)
        weight = Tensor(None, shape=(4,32), is_realized=False)
        bias = Tensor(None, shape=(32,32), is_realized=False)
        y = x @ weight 
        z = y + bias
        g = Graph(z)
        program = g.export(inputs=(x,))
        snapshots = program.life_snapshots
        """
        add 5290 (32, 32)
            - 97d0 matmul (32, 32) [nan]
            | - 8b90 (32, 4) [nan]
            | - 5bd0 (4, 32) [nan]
            - 68d0 (32, 32) [nan]
        ->  [(['97d0'], ['97d0']), (['97d0', '5290'], ['5290'])]
        """
        self.assertTrue(get_id(y)==(snapshots[0][0][0]))
        self.assertTrue(get_id(y)==(snapshots[0][1][0]))

        self.assertTrue(get_id(y)==(snapshots[1][0][0]),f"{get_id(y),(snapshots[1][0][0])}")
        self.assertTrue(get_id(z)==(snapshots[1][0][1]))

        self.assertTrue(get_id(z)==(snapshots[1][1][0]))

if __name__ == "__main__":
    unittest.main()