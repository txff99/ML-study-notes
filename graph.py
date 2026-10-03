from tensor import Tensor
from typing import Callable
from backend import Backend
"""
how to rewrite graph
1. traverse
2. match -> replace
"""

class Graph:
    def __init__(self, tensor:Tensor):
         self.root = tensor
    
    def printAST(self):
        if self.root is not None:
            print(hex(id(self.root))[-4:], self.root.shape)
            self.root.function.printAST()

    def toposort(self) -> list[Tensor]:
        visited = set()
        linear_tensors = []
        def dfs(tensor: Tensor):
            if tensor in visited: return
            visited.add(tensor)    
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
            linear_tensors.append(tensor)
        dfs(self.root)
        return linear_tensors

    def rewrite(self, passes: list[Callable]|Callable, args:list[tuple]=None):
        visited = set()
        def dfs(tensor: Tensor):
            if tensor in visited: return
            visited.add(tensor) 
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
            self.apply_passes(tensor, passes, args)
        dfs(self.root)        

    def apply_passes(self, tensor: Tensor, passes: list[Callable]|Callable, args:list[tuple]|tuple=None):
        if callable(passes): passes = [passes]
        if args is not None and isinstance(args, tuple): 
            args = list[args]
            assert len(passes)==len(args), "number of args should match passes"
        for i,_ in enumerate(passes):
            if args is not None:
                passes[i](tensor, args[i])
            else:
                passes[i](tensor)
    
    def is_realizable(self):
        edges = []
        visited = set()
        def dfs(tensor: Tensor):
            if tensor in visited: return
            visited.add(tensor) 
            if tensor.function is None: 
                edges.append(tensor)
                return
            for mem in tensor.function.parents:
                dfs(mem)
        dfs(self.root)
        return all([e.is_realized for e in edges])

def add_contiguous_before_ari(tensor: Tensor, args=None):
    arithmetic_funcs = {"add","sub","matmul","maximum","mse","max","mul","div","exp","sum","sqrt"}
    if tensor.function is not None and tensor.function.name in arithmetic_funcs:
        # replace tensor parent with parent.contiguous
        for i,_ in enumerate(tensor.function.parents):
            tensor.function.parents[i] = tensor.function.parents[i].contiguous()

def constant_folding(tensor: Tensor, args=None):
    if tensor.function is None: return
    if all([par.is_realized for par in tensor.function.parents]):
        tensor.realize()
