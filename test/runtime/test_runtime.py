import numpy as np
import sys
import unittest

sys.path.append("../..")
from tensor import Tensor
from graph import Graph
class TestNN(unittest.TestCase):
    def test_replay(self):
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