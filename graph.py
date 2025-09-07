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
            self.apply_passes(tensor, passes, args)
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
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

def alloc_tensor_pass(tensor: Tensor, args: tuple[Backend]):
    backend = args
    assert args is not None, "alloc_tensor_pass needs to specify backend"
    assert isinstance(backend,Backend), "alloc_tensor_rule args not match"
    tensor.backend = backend
    visited.add(tensor)
    backend.alloc(tensor)
    if tensor.is_realized == True:
        backend.copyHTOD(tensor)

def add_contiguous_before_ari(tensor: Tensor, args=None):
    arithmetic_funcs = {"add","sub","matmul","mse","max","mul","div","exp","sum","sqrt"}
    if tensor.function is not None and tensor.function.name in arithmetic_funcs:
        # replace tensor parent with parent.contiguous
        for i,_ in enumerate(tensor.function.parents):
            tensor.function.parents[i] = tensor.function.parents[i].contiguous()
        
