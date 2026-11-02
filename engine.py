from __future__ import annotations
from tensor import Op, OpType, Tensor
import numpy as np
from typing import List
from backend import Backend
import ctypes
from dataclasses import dataclass

"""
engine used to run the ops
"""

@dataclass
class Block:
    addr: int
    size: int
    left: Block = None
    right: Block = None
    occupied: bool = False
    empty: bool = True
    def __repr__(self):
        return f"(addr={self.addr}, size={self.size}, occupied={self.occupied}, empty={self.empty})"

class MemPool:
    def __init__(self, total_byte=1024):
        # total byte and minimum blk size need to be exponent of 2
        def is_power_of_two(n): return n > 0 and (n & (n - 1)) == 0
        assert is_power_of_two(total_byte), "total byte should be exponent of 2"
        self.buffer = np.empty(total_byte, dtype=np.uint8)
        self.total_size = total_byte
        self.minimum_blk_size = 16
        self.root = Block(addr=0, size=total_byte)
    
    def printState(self):
        def dfs(blk, level=""):
            if blk is None: return 
            print(level,blk)
            dfs(blk.left, level+"   ")
            dfs(blk.right, level+"   ")
        dfs(self.root)

    def alloc(self, size, dtype=np.float32):
        assert size <= self.total_size, f"size={size}, total_size={self.total_size}"
        def prop_occupy_state(blk):
            if (blk.left is not None and blk.left.occupied) and \
                (blk.right is not None and blk.right.occupied): 
                blk.occupied=True
        def dfs(blk):
            if blk.occupied: return None
            if (size <= blk.size and size > blk.size //2) or blk.size==self.minimum_blk_size:
                if not blk.empty: return None
                blk.empty=False
                blk.occupied=True
                return (blk.addr, self.buffer[blk.addr: blk.addr + size].view(dtype))
            blk.empty=False
            if blk.left is None: blk.left = Block(addr=blk.addr, size=blk.size//2)
            if not blk.left.occupied: 
                ret = dfs(blk.left) 
                if ret is not None: 
                    prop_occupy_state(blk)
                    return ret

            if blk.right is None: blk.right = Block(addr=blk.addr + blk.size//2, size=blk.size//2)
            if not blk.right.occupied: 
                ret = dfs(blk.right) 
                if ret is not None: 
                    prop_occupy_state(blk)
                    return ret

        ret = dfs(self.root)
        if ret is None: print(f"alloc {size} fail")
        return ret
    
    def dealloc(self, addr):
        def dfs(blk):
            if addr >= blk.addr + blk.size//2:
                if blk.right: dfs(blk.right)
            else:
                if blk.left: dfs(blk.left)
            
            blk.occupied = False

            if blk.left is not None and blk.left.empty:
                blk.left = None
            if blk.right is not None and blk.right.empty:
                blk.right = None
            if blk.left is None and blk.right is None:
                blk.empty = True
            
        dfs(self.root)

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
