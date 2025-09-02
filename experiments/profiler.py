import time
import functools


def profile(func):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"[PROFILE] func {func.__name__}(...) - execution time: {elapsed_time:.4f}s")
        
        return result
    
    return wrapper