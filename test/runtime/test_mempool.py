import numpy as np
import sys
import unittest

sys.path.append("../..")
from engine import MemPool
class TestRuntime(unittest.TestCase):
    def test_mempool_simple1(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(32)
        pool.dealloc(0)
        self.assertEqual(bytes1[0],0)
    
    def test_mempool_small(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(8)        
        pool.dealloc(0)
        self.assertEqual(bytes1[0],0)

    def test_mempool_simple3(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(32)
        bytes2 = pool.alloc(128)
        self.assertTrue(bytes1[0]==0 and bytes2[0]==128)
        pool.dealloc(0)
        self.assertFalse(pool.root.empty)
        pool.dealloc(128)
        self.assertTrue(pool.root.empty)
    
    def test_mempool_complex(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(1024)        
        pool.dealloc(0)
        self.assertFalse(pool.root.occupyied)
        bytes1 = pool.alloc(8)        
        self.assertEqual(bytes1[0],0)
        pool.dealloc(0)
        self.assertFalse(pool.root.occupyied)


if __name__ == "__main__":
    unittest.main()