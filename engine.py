from tensor import Op, OpType, Tensor
import numpy as np
from typing import List
from backend import Backend
import ctypes

"""
engine used to run the ops
"""

class Engine:
    def __init__(self, backend: Backend, ops: List[Op] = None, mode="once"):
        self.backend = backend
        self.ops = ops
        self.mode=mode

    def run(self):
        assert self.ops is not None , "ops should be applied before run"
        for op in self.ops:
            optype: OpType = op.optype
            srcs: List[Tensor] = op.srcs
            dst: Tensor = op.dst
            assert optype in self.backend.supported_ops, f"{optype} not supported by backend"
            self.backend.execute(op)
            if self.mode == "once": dst.is_realized = True
