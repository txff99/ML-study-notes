from tensor import Op, OpType, Tensor
import numpy as np
from typing import List
from backend import Backend
import ctypes

"""
engine used to run the ops
"""

class Engine:
    def __init__(self, backend: Backend, ops: List[Op] = None):
        self.backend = backend
        self.ops = ops

    def run(self):
        assert self.ops is not None , "ops should be applied before run"
        for op in self.ops:
            optype: OpType = op.optype
            srcs: List[Tensor] = op.srcs
            dst: Tensor = op.dst
            if dst.is_realized == True: continue
            assert optype in self.backend.supported_ops, "op not supported by backend"
            self.backend.execute(op)
            dst.is_realized = True
