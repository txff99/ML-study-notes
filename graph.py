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
    def __init__(self, inputs, ops, outputs, backend=None, life_snapshots=None, enable_memory_plan=False, memory_pool_size=1024):
        self.inputs = inputs
        self.ops: List[Op] = ops 
        self.outputs = outputs
        if backend is None:
            backend = CPU()
        self.backend = backend
        self.life_snapshots = life_snapshots
        self.enable_memory_plan = enable_memory_plan
        if enable_memory_plan:
            self.backend.set_inplace(True)
            self.compile_memory_plan(pool_size=memory_pool_size)

    def print_life_snapshots(self):
        def get_id(x):
            return hex(id(x))[-4:]
        if self.life_snapshots is not None:
            print_str = ""
            for alloc, dealloc in self.life_snapshots:
                print_str += f"(alloc: {get_id(alloc[0])}, dealloc: {[get_id(de) for de in dealloc]}) "
            print(print_str)

    def compile_memory_plan(self, pool_size=1024):
        assert self.life_snapshots is not None
        from engine import MemPool
        from util import get_tensor_size
        mempool = MemPool(total_byte=pool_size)
        addr_tracker = {}
        max_memory = 0
        assert len(self.life_snapshots) == len(self.ops)
        for (idx, snapshot) in enumerate(self.life_snapshots):
            alloc, dealloc = snapshot
            tensor_alloc = alloc[0]
            alloc_size = get_tensor_size(tensor_alloc)
            bytes = mempool.alloc(alloc_size, dtype=tensor_alloc.dtype)
            max_memory += alloc_size
            assert bytes is not None, f"memory plan failed, max_memory={max_memory}, pool_size={pool_size}"
            addr_tracker[tensor_alloc] = bytes[0]
            # bind memory to tensor's data
            self.ops[idx].dst.data = bytes[1]
            tensor_deallocs = dealloc
            for tensor_dealloc in tensor_deallocs:
                dealloc_size = get_tensor_size(tensor_dealloc)
                mempool.dealloc(addr_tracker[tensor_dealloc])
                max_memory -= dealloc_size

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
        from util import get_tensor_size
        ref_tracker = {}
        life_snapshots = []
        visited = set()
        def get_tensor_refcount(tensor: Tensor):
            if tensor.function is None: return
            # tensor_id = hex(id(tensor))[-4:]
            if tensor not in ref_tracker: ref_tracker[tensor] = 0
            ref_tracker[tensor] += 1
            if tensor in visited: return
            visited.add(tensor)    
            for mem in tensor.function.parents:
                get_tensor_refcount(mem)

        for root in self.roots:
            get_tensor_refcount(root)
        
        visited.clear()
        def get_tensor_snapshot(tensor: Tensor):
            if tensor.function is None: return
            if tensor in visited: return
            visited.add(tensor)  
            for mem in tensor.function.parents:
                get_tensor_snapshot(mem)

            alloc = [tensor]
            dealloc = []
            for mem in tensor.function.parents:
                if mem in ref_tracker:
                    ref_tracker[mem]-=1
                    if ref_tracker[mem]==0:
                        dealloc.append(mem)
            life_snapshots.append((alloc,dealloc))
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

    def export(self, inputs:List[Tensor], enable_memory_plan=False, memory_pool_size=1024):
        ops = self.lower()
        snapshots = None
        if enable_memory_plan:
            snapshots = self.get_lifetime_snapshots()
        return Program(ops=ops, inputs=inputs, outputs=self.roots, life_snapshots=snapshots, enable_memory_plan=enable_memory_plan, memory_pool_size=memory_pool_size)

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