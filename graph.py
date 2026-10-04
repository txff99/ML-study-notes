from tensor import Tensor
from typing import Callable
from backend import Backend
"""
how to rewrite graph
1. traverse
2. match -> replace
"""

def all_parents_evaluated(tensor:Tensor):
    return all([par.is_realized for par in tensor.function.parents])

def is_function_arithmetic(tensor:Tensor):
    arithmetic_funcs = {"add","sub","matmul","maximum","mse","max","mul","div","exp","sum","sqrt"}
    return tensor.function is not None and tensor.function.name in arithmetic_funcs

def is_function_add_sub(tensor:Tensor):
    return tensor.function is not None and tensor.function.name in {"add","sub"}

def is_function_mul_div(tensor:Tensor):
    return tensor.function is not None and tensor.function.name in {"add","sub"}

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
    if is_function_arithmetic(tensor):
        # replace tensor parent with parent.contiguous
        for i,_ in enumerate(tensor.function.parents):
            tensor.function.parents[i] = tensor.function.parents[i].contiguous()

def constant_folding(tensor: Tensor, args=None):
    # pattern match 1: if all parents realized, realize this tensor
    if all_parents_evaluated(tensor):
        tensor.realize()
        # remove all parents.
        tensor.function = None
        return

    # pattern match2: fold arithmetic op add/sub
    if is_function_add_sub(tensor) and is_only_one_src_realized(tensor):
        r1, r2 = tensor.function.parents
        r = tensor
        rr = r1 if r1.is_realized else r2
        r_ = r2 if r1.is_realized else r1
        no_op = Tensor(0) 
        if is_function_add_sub(r_) and is_only_one_src_realized(r_):
            r_1, r_2 = r_.function.parents
            rrr = r_1 if r_1.is_realized else r2
            r__ = r_2 if r_1.is_realized else r1
            func2 = r_.function
            func2.parents = [rrr, no_op] if r_1.is_realized else [no_op, rrr]
            folding_tensor2 = Tensor(None, shape=rrr.shape, function=func2,is_realized=False)
            func1 = r.function
            func1.parents = [rr, folding_tensor2] if r1.is_realized else [folding_tensor2, rr]
            folding_tensor1 = Tensor(None, shape=rr.shape, function=func1, is_realized=False)
            folding_tensor1.realize()
            # special case: (folding1 + (folding2 - r__)) where r__ is negative
            r.function.parents = [folding_tensor1, r__] if r1.is_realized else [r__, folding_tensor1]
            if r_.function.name == "sub" and r.function.name=="add" and r_1.is_realized:
                r.function.name = "sub"
            # ((folding1 - r__) + folding2)
                if r2.is_realized:
                    r.function.parents = [folding_tensor1, r__]
            return
