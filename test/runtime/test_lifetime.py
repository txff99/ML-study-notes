import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph
class TestRuntime(unittest.TestCase):
    def test_single(self):
        x = Tensor(None, shape=(32,4), is_realized=False)
        weight = Tensor(None, shape=(4,32), is_realized=False)
        bias = Tensor(None, shape=(32,32), is_realized=False)
        y = x @ weight 
        z = y + bias
        g = Graph(z)
        program = g.export(inputs=(x,),enable_memory_plan=True,memory_pool_size=65536)
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

    def test_multi(self):
        x = Tensor(None, shape=(32,4), is_realized=False)
        weight = Tensor(None, shape=(4,32), is_realized=False)
        bias = Tensor(None, shape=(32,32), is_realized=False)
        y = x @ weight 
        z = y + bias
        w = z + bias
        g = Graph(z, w)
        program = g.export(inputs=(x,),enable_memory_plan=True,memory_pool_size=65536)
        snapshots = program.life_snapshots
        """
        add d0d0 (32, 32)
        - 9e90 matmul (32, 32) [nan]
        | - 88d0 (32, 4) [nan]
        | - 7190 (4, 32) [nan]
        - 7250 (32, 32) [nan]
        add d1d0 (32, 32)
        - d0d0 add (32, 32) [nan]
        | - 9e90 matmul (32, 32) [nan]
        |    | - 88d0 (32, 4) [nan]
        |    | - 7190 (4, 32) [nan]
        | - 7250 (32, 32) [nan]
        - 7250 (32, 32) [nan]
        (alloc: 9e90, dealloc: []) (alloc: d0d0, dealloc: ['9e90']) (alloc: d1d0, dealloc: []) 
        """
        alloc1 = snapshots[0][0]
        dealloc1 = snapshots[0][1]
        self.assertTrue(y==(alloc1[0]))
        self.assertTrue(len(dealloc1)==0)
        alloc2 = snapshots[1][0]
        dealloc2 = snapshots[1][1]
        self.assertTrue(z==(alloc2[0]))
        self.assertTrue(y==(dealloc2[0]))
        alloc3 = snapshots[2][0]
        dealloc3 = snapshots[2][1]
        self.assertTrue(w==(alloc3[0]))
        self.assertTrue(len(dealloc3)==0)

    def test_exported_program(self):
        np.random.seed(1)
        src = np.random.rand(32,4)
        raw1 = np.random.rand(4, 32) 
        raw2 = np.random.rand(32, 32)  
        inp = Tensor(src)
        x = Tensor(None, shape=(32,4), is_realized=False)
        weight = Tensor(raw1)
        bias = Tensor(raw2)
        y = x @ weight + bias
        g = Graph(y)
        program = g.export(inputs=(x,),enable_memory_plan=True,memory_pool_size=65536)
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