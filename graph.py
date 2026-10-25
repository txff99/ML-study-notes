from tensor import Tensor, Op, FUNCTION_TO_OPTYPE
from typing import Callable
from backend import Backend, CPU
from typing import List, Tuple
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

class Program:
    def __init__(self, inputs, ops, outputs, backend=CPU(), life_snapshots=None):
        self.inputs = inputs
        self.ops: List[Op] = ops 
        self.outputs = outputs
        self.backend = backend
        self.life_snapshots = life_snapshots

    def run(self, inputs):
        from engine import Engine
        from util import bind_tensor_dyshape
        assert len(inputs)==len(self.inputs) and \
         all([inp.is_realized and inp.shape for inp in inputs]) and \
         all([inp1.shape == inp2.shape for inp1, inp2 in zip(inputs,self.inputs) if not inp2.has_dyshape])
        # copy inputs into input buffer
        for dst, src in zip(self.inputs, inputs):
            if dst.has_dyshape:
                bind_tensor_dyshape(dst, src)
            dst.data = src.data
        engine = Engine(self.backend, self.ops, mode="replay")
        engine.run()
        for output in self.outputs:
            output.is_realized = True
        return self.outputs

class Graph:
    def __init__(self, *tensors:List[Tensor]):
         self.roots = list(tensors)
    
    def printAST(self):
        for root in self.roots:
            if root is not None:
                print(root.function.name, hex(id(root))[-4:], root.shape)
                root.function.printAST()

    def toposort(self) -> list[Tensor]:
        visited = set()
        linear_tensors = []
        def dfs(tensor: Tensor):
            if tensor.function is None: return
            if tensor in visited: return
            visited.add(tensor)    
            for mem in tensor.function.parents:
                dfs(mem)
            linear_tensors.append(tensor)
        for root in self.roots:
            dfs(root)
        return linear_tensors
    
    def lower(self) -> List[Op]: 
        # lower a dag to ops
        from util import dyint_in_tensor
        from tensor import sint
        self.rewrite(add_contiguous_before_ari)
        tensors = self.toposort()
        ops = []
        for tensor in tensors:
            tensor_op = []
            if tensor.has_dyshape:
                tensor_op = [Op(FUNCTION_TO_OPTYPE[dyint.function.name], dyint.function.parents, tensor) for dyint in dyint_in_tensor(tensor) if isinstance(dyint.data, sint)]
            tensor_op.append(Op(FUNCTION_TO_OPTYPE[tensor.function.name], tensor.function.parents, tensor))
            ops.extend(tensor_op)
        return ops

    def get_lifetime_snapshots(self):
        ref_tracker = {}
        life_snapshots = []
        life_snapshot = []
        visited = set()
        def get_tensor_refcount(tensor: Tensor):
            if tensor.function is None: return
            tensor_id = hex(id(tensor))[-4:]
            if tensor_id not in ref_tracker: ref_tracker[tensor_id] = 0
            ref_tracker[tensor_id] += 1
            if tensor in visited: return
            visited.add(tensor)    
            for mem in tensor.function.parents:
                get_tensor_refcount(mem)

        for root in self.roots:
            get_tensor_refcount(root)
        
        visited.clear()
        def get_tensor_snapshot(tensor: Tensor):
            if tensor.function is None: return
            tensor_id = hex(id(tensor))[-4:]
            if tensor in visited: return
            visited.add(tensor)  
            for mem in tensor.function.parents:
                get_tensor_snapshot(mem)

            life_snapshot.append(tensor_id)
            pre = life_snapshot.copy()
            for mem in tensor.function.parents:
                mem_id = hex(id(mem))[-4:]
                if mem_id in ref_tracker:
                    ref_tracker[mem_id]-=1
                    if ref_tracker[mem_id]==0:
                        life_snapshot.remove(mem_id)
            post = life_snapshot.copy()
            life_snapshots.append((pre,post))
        for root in self.roots:
            get_tensor_snapshot(root)
        return life_snapshots

    def rewrite(self, passes: list[Callable]|Callable):
        if callable(passes): passes = [passes]
        for p in passes:
            self.apply_pass(self.roots, p)        

    def apply_pass(self, tensors: List[Tensor], p: Callable):
        visited = set()
        def dfs(tensor: Tensor):
            if tensor in visited: return
            visited.add(tensor) 
            if tensor.function is None: return
            for mem in tensor.function.parents:
                dfs(mem)
            p(tensor)
        for tensor in tensors:
            dfs(tensor)
    
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
        for root in self.roots:
            dfs(root)  
        return all([e.is_realized for e in edges])

    def export(self, inputs:List[Tensor]):
        ops = self.lower()
        snapshots = self.get_lifetime_snapshots()
        return Program(ops=ops, inputs=inputs, outputs=self.roots, life_snapshots=snapshots)

def canonicalize(tensor:Tensor):
    c1 = Tensor(1, is_fake=True)
    x = Tensor(None, shape=(), is_realized=False, is_fake=True)
    RULES = [
        (c1 + x, lambda c1, x: x + c1),
        (c1 * x, lambda c1, x: x * c1),
    ]

    for pattern, replacement in RULES:
        bindings = match(tensor, pattern)
        if bindings is not None:
            new_tensor = replacement(*bindings)
            tensor.replace(new_tensor)
            return

def add_contiguous_before_ari(tensor: Tensor, args=None):
    arithmetic_funcs = {"add","sub","matmul","maximum","mse","max","mul","div","exp","sum","sqrt", "where"}
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

    w = Tensor(None, shape=(3,4), is_realized=True, is_fake=True)
    v = Tensor(None, shape=(4,1), is_realized=False, is_fake=True)
    b = Tensor(None, shape=(3,1), is_realized=True, is_fake=True)
    s = Tensor(None, shape=(3,1), is_realized=True, is_fake=True)
    
    FOLD_RULES = [
        # add/sub
        ((x + c1) + c2, lambda x, c1, c2: x + (c1 + c2)),
        (c1 - (x + c2), lambda c1, x, c2 : (c1 + c2) - x),
        ((x + c1) - c2, lambda x, c1, c2: x + (c1 - c2)),
        (c1 - (x - c2), lambda c1, x, c2 : (c1 - c2) - x),
        ((c1 - x) - c2, lambda c1, x, c2: (c1 - c2) - x),
        ((x - c1) - c2, lambda x, c1, c2: x - (c1 + c2)),
        (c1 - (c2 - x), lambda c1, c2, x: (c1 - c2) + x),
        ((c1 - x) + c2, lambda c1, x, c2:  (c1 + c2) - x),
        ((x - c1) + c2, lambda x, c1, c2: x + (c2 - c1)),

        # mul/div
        ((x * c1) * c2, lambda x, c1, c2: x + (c1 * c2)),
        (c1 / (x * c2), lambda c1, x, c2 : (c1 * c2) / x),
        ((x * c1) / c2, lambda x, c1, c2: x * (c1 / c2)),
        (c1 / (x / c2), lambda c1, x, c2 : (c1 / c2) / x),
        ((c1 / x) / c2, lambda c1, x, c2: (c1 / c2) / x),
        ((x / c1) / c2, lambda x, c1, c2: x / (c1 * c2)),
        (c1 / (c2 / x), lambda c1, c2, x: (c1 / c2) * x),
        ((c1 / x) * c2, lambda c1, x, c2:  (c1 * c2) / x),
        ((x / c1) * c2, lambda x, c1, c2: x * (c2 / c1)),

        # matmul
        ((w @ v + b) / s, lambda w, v, b, s : ((w/s) @ v + b / s)),
        ((w @ v) / s, lambda w, v, s : (w/s) @ v),
    ]

    for pattern, replacement in FOLD_RULES:
        bindings = match(tensor, pattern)
        if bindings is not None:
            new_tensor = replacement(*bindings)
            tensor.replace(new_tensor)
            return

def constant_folding_pass(tensor:Tensor):
    return [
        (canonicalize, tensor),
        (constant_folding_pass, tensor),
        (constant_realize, tensor),
    ]