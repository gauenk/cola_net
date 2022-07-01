import torch as th

def print_gpu_stats(verbose,name):
    fmt_all = "[%s] Memory Allocated: %2.3f"
    fmt_res = "[%s] Memory Reserved: %2.3f"
    if verbose:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        mem = th.cuda.memory_allocated() / 1024**3
        print(fmt_all % (name,mem))
        mem = th.cuda.memory_reserved() / 1024**3
        print(fmt_res % (name,mem))

def reset_peak_gpu_stats():
    th.cuda.reset_max_memory_allocated()

def print_peak_gpu_stats(verbose,name,reset=True):
    fmt = "[%s] Peak Memory: %2.3f"
    if verbose:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        mem = th.cuda.max_memory_allocated(0)
        mem_gb = mem / (1024**3)
        print(fmt % (name,mem))
        print("Max Mem (GB): ",mem_gb)
        if reset: th.cuda.reset_peak_memory_stats()



