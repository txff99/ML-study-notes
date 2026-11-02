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
    
    def test_mempool_simple2(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(32)
        bytes2 = pool.alloc(128)
        self.assertTrue(bytes1[0]==0 and bytes2[0]==128)
        pool.dealloc(0)
        self.assertFalse(pool.root.empty)
        pool.dealloc(128)
        self.assertTrue(pool.root.empty)
    
    def test_mempool_small(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(8)        
        pool.dealloc(0)
        self.assertEqual(bytes1[0],0)

    def test_mempool_big(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(36)        
        pool.dealloc(0)
        self.assertEqual(bytes1[0],0)
    
    def test_mempool_complex(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(1024)        
        pool.dealloc(0)
        self.assertFalse(pool.root.occupied)
        bytes1 = pool.alloc(8)        
        self.assertEqual(bytes1[0],0)
        pool.dealloc(0)
        self.assertFalse(pool.root.occupied)
    
    def test_mempool_repeat(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(32)  
        pool.dealloc(0)
        pool.alloc(16)
        pool.dealloc(0)
        bytes = pool.alloc(128)
        self.assertTrue(bytes[0]==0)

    def test_mempool_occup_prop(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(32)  
        bytes2 = pool.alloc(32)
        bytes3 = pool.alloc(64)
        self.assertTrue(pool.root.left.left.left.occupied) # 128
        self.assertTrue(pool.root.left.left.left.left.occupied) # 64
    
    def test_mempool_dealloc_prop(self):
        pool = MemPool(1024)
        bytes1 = pool.alloc(32)  
        bytes2 = pool.alloc(32)
        bytes3 = pool.alloc(64)
        pool.dealloc(0)
        self.assertFalse(pool.root.left.left.left.left.occupied) # 64
        pool.dealloc(32)
        self.assertTrue(pool.root.left.left.left.left is None) # 64
        pool.dealloc(64)
        self.assertTrue(pool.root.empty)

    @unittest.expectedFailure
    def test_mempool_overflow(self):
        pool = MemPool(1024)
        bytes = pool.alloc(1025)
        self.assertTrue(bytes is None)

    def test_mempool_overflow2(self):
        pool = MemPool(1024)
        pool.alloc(512)
        pool.alloc(512)
        bytes = pool.alloc(1)
        self.assertTrue(bytes is None)


if __name__ == "__main__":
    unittest.main()