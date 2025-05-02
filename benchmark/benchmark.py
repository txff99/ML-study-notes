import numpy as np
import sys
import time

sys.path.append("..")
from tensor import Tensor

def run(t: Tensor):
    t.evaluate(cleanup_after_eval=True)

if __name__ == "__main__":
    size_m = 2048
    size_n = 2048
    size_k = 256
    GFLOP = 2*size_m*size_n*size_k*1e-9
    cpu_time = 0
    gpu_time = 0
    epoch = 100
    for i in range(epoch):
        A = Tensor(data=np.random.rand(size_m, size_k))
        B = Tensor(data=np.random.rand(size_k, size_n))
        C_cpu = A @ B
        
        start_cpu = time.time()
        run(C_cpu)
        cpu_time += time.time()-start_cpu
        
        C_gpu = A @ B
        C_gpu.toGPU()
        start_gpu = time.time()
        run(C_gpu)
        gpu_time += time.time()-start_gpu
    
    print(f"cpu perf: {epoch*GFLOP/cpu_time:.2f} GFOPS")
    print(f"gpu perf: {epoch*GFLOP/gpu_time:.2f} GFOPS")