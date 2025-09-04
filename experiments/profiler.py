import time
import functools
import torch


def profile(func):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_reserved_mem = torch.cuda.memory_reserved() / 1024**3
        result = func(*args, **kwargs)
        end_time = time.time()
        end_reserved_mem = torch.cuda.memory_reserved() / 1024**3
        elapsed_time = end_time - start_time
        reserved_mem_added = end_reserved_mem - start_reserved_mem
        
        print(f"[PROFILE] func {func.__name__}(...) execution time: {elapsed_time:.4f}s gpu mem reserved: {reserved_mem_added:.1f}GB")
        
        return result
    
    return wrapper