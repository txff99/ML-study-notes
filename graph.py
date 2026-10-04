from tensor import Tensor
from typing import Callable
from backend import Backend
"""
how to rewrite graph
1. traverse
2. match -> replace
"""

def match(source_tensor:Tensor, target_tensor):
    visited = set()
    bindings = []
    def dfs(source: Tensor, target: Tensor):
        if source in visited or target in visited: return True
        visited.add(target) 
        visited.add(source)
        if target.is_fake:
            bindings.append(source)
        if source.is_realized ^ target.is_realized: return False
        if target.function is None: 
            return True
        if source.function is None: return False 
        if source.function.name != target.function.name: return False
        if len(source.function.parents) != len(target.function.parents): return False
        for s, t in zip(source.function.parents, target.function.parents):
            if not dfs(s, t): return False
        return True
    return bindings if dfs(source_tensor, target_tensor) else None

def all_parents_evaluated(tensor:Tensor):
    return all([par.is_realized for par in tensor.function.parents])

def is_function_arithmetic(tensor:Tensor):
    arithmetic_funcs = {"add","sub","matmul","maximum","mse","max","mul","div","exp","sum","sqrt"}
    return tensor.function is not None and tensor.function.name in arithmetic_funcs

def is_function_add_sub(tensor:Tensor):
    return tensor.function is not None and tensor.function.name in {"add","sub"}

def is_function_mul_div(tensor:Tensor):
    return tensor.function is not None and tensor.function.name in {"mul","div"}

def is_function_arith_binop(tensor:Tensor):
    return is_function_add_sub(tensor) or is_function_mul_div(tensor)

def is_function_matmul(tensor:Tensor):
    return tensor.function is not None and tensor.function.name == "matmul"

def is_only_one_src_realized(tensor:Tensor):
    if tensor.function is None: return False
    assert len(tensor.function.parents)==2
    r1, r2 = tensor.function.parents
    return r1.is_realized ^ r2.is_realized

class Graph:
    def __init__(self, tensor:Tensor):
         self.root = tensor
    
    def printAST(self):
        if self.root is not None:
            print(self.root.function.name, hex(id(self.root))[-4:], self.root.shape)
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
    if is_function_arithmetic(tensor):
        # replace tensor parent with parent.contiguous
        for i,_ in enumerate(tensor.function.parents):
            tensor.function.parents[i] = tensor.function.parents[i].contiguous()

def constant_realize(tensor: Tensor, args=None):
    # todo: make sure no subgraph has tensor requires grad
    if all_parents_evaluated(tensor):
        tensor.realize()
        # remove all parents.
        tensor.function = None
        return

def constant_folding(tensor: Tensor, args=None):
    c1 = Tensor(1, is_fake=True)
    c2 = Tensor(1, is_fake=True) 
    x = Tensor(None, shape=(), is_realized=False, is_fake=True)
    
    FOLD_RULES = [
        # add/sub
        (c1 + (x + c2), lambda c1, x, c2 : x + (c1 + c2)),
        ((c1 + x) + c2, lambda c1, x, c2: x + (c1 + c2)),
        ((x + c1) + c2, lambda x, c1, c2: x + (c1 + c2)),
        (c1 + (c2 + x), lambda c1, c2, x: x + (c1 + c2)),
        (c1 - (x + c2), lambda c1, x, c2 : (c1 + c2) - x),
        ((c1 + x) - c2, lambda c1, x, c2: x + (c1 - c2)),
        ((x + c1) - c2, lambda x, c1, c2: x + (c1 - c2)),
        (c1 - (c2 + x), lambda c1, c2, x: x + (c1 - c2)),
        (c1 - (x - c2), lambda c1, x, c2 : (c1 - c2) - x),
        ((c1 - x) - c2, lambda c1, x, c2: (c1 - c2) - x),
        ((x - c1) - c2, lambda x, c1, c2: x - (c1 + c2)),
        (c1 - (c2 - x), lambda c1, c2, x: (c1 - c2) + x),
        (c1 + (x - c2), lambda c1, x, c2 : x + (c1 - c2)),
        ((c1 - x) + c2, lambda c1, x, c2:  (c1 + c2) - x),
        ((x - c1) + c2, lambda x, c1, c2: x + (c2 - c1)),
        (c1 + (c2 - x), lambda c1, c2, x: (c1 + c2) - x),

        # mul/div
        (c1 * (x * c2), lambda c1, x, c2 : x * (c1 * c2)),
        ((c1 * x) * c2, lambda c1, x, c2: x * (c1 * c2)),
        ((x * c1) * c2, lambda x, c1, c2: x + (c1 * c2)),
        (c1 * (c2 * x), lambda c1, c2, x: x + (c1 * c2)),
        (c1 / (x * c2), lambda c1, x, c2 : (c1 * c2) / x),
        ((c1 * x) / c2, lambda c1, x, c2: x * (c1 / c2)),
        ((x * c1) / c2, lambda x, c1, c2: x * (c1 / c2)),
        (c1 / (c2 * x), lambda c1, c2, x: x * (c1 / c2)),
        (c1 / (x / c2), lambda c1, x, c2 : (c1 / c2) / x),
        ((c1 / x) / c2, lambda c1, x, c2: (c1 / c2) / x),
        ((x / c1) / c2, lambda x, c1, c2: x / (c1 * c2)),
        (c1 / (c2 / x), lambda c1, c2, x: (c1 / c2) * x),
        (c1 * (x / c2), lambda c1, x, c2 : x * (c1 / c2)),
        ((c1 / x) * c2, lambda c1, x, c2:  (c1 * c2) / x),
        ((x / c1) * c2, lambda x, c1, c2: x * (c2 / c1)),
        (c1 * (c2 / x), lambda c1, c2, x: (c1 * c2) / x),
    ]

    for pattern, replacement in FOLD_RULES:
        bindings = match(tensor, pattern)
        if bindings is not None:
            new_tensor = replacement(*bindings)
            tensor.replace(new_tensor)
            return
