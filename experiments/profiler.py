from collections import defaultdict
import time
import functools
import psutil
import torch
import atexit

class Profiler():
    stats = defaultdict(lambda: 
                        {"calls": 0, 
                         "time": 0.0, 
                         "gpu_mem": 0.0,
                         "cpu_mem": 0.0,
                         })

    @classmethod
    def profile(cls, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_reserved_gpu_mem = torch.cuda.memory_reserved() / 1024**3
            start_unavailable_cpu_mem_pct = psutil.virtual_memory().percent

            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.time()
                end_reserved_gpu_mem = torch.cuda.memory_reserved() / 1024**3
                end_unavailable_cpu_mem_pct = psutil.virtual_memory().percent

                elapsed_time = end_time - start_time
                reserved_mem_added = end_reserved_gpu_mem - start_reserved_gpu_mem
                cpu_mem_added = end_unavailable_cpu_mem_pct - start_unavailable_cpu_mem_pct

                cls.stats[func.__name__]["calls"] += 1
                cls.stats[func.__name__]["time"] += elapsed_time
                cls.stats[func.__name__]["gpu_mem"] += reserved_mem_added   
                cls.stats[func.__name__]["cpu_mem"] += cpu_mem_added   

                print(f"[PROFILE] func {func.__name__}(...)"
                      f"    execution time: {elapsed_time:.4f}s "
                      f"    gpu mem reserved: {reserved_mem_added:.1f}GB")
                
        return wrapper
    
    @classmethod
    def print_stats(self):
        print(f"\n\n=========Profiler Stats=========")

        for func, stats in self.stats.items():
            avg_time = stats["time"] / stats["calls"] if stats["calls"] else 0.0
            avg_gpu_mem = stats["gpu_mem"] / stats["calls"] if stats["calls"] else 0.0
            avg_cpu_mem = stats["cpu_mem"] / stats["calls"] if stats["calls"] else 0.0
            
            print(
                f"{func}: calls={stats['calls']}    "
                f"total_time={stats['time']:.4f}s avg_time={avg_time:.4f}s    "
                f"avg_gpu_mem={avg_gpu_mem:.3f}GB    "
                f"avg_cpu_mem={avg_cpu_mem:.3f}%    "
            )
        print(f"===============================\n\n")

# Create alias so `@profile decorator works `
profile = Profiler.profile 
atexit.register(Profiler.print_stats)