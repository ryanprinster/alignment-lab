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
                         "gpu_mem_res": 0.0,
                         "gpu_mem_peak": 0.0,
                         "gpu_mem_peak_delta": 0.0,
                         "gpu_mem_final_delta": 0.0,
                         "cpu_mem": 0.0,
                         })

    @classmethod
    def profile(cls, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get class name
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                func_name = f"{class_name}.{func.__name__}"
            else:
                func_name = func.__name__
            
            start_time = time.time()
            start_reserved_gpu_mem = torch.cuda.memory_reserved() / 1024**3
            start_unavailable_cpu_mem_pct = psutil.virtual_memory().percent
            torch.cuda.reset_peak_memory_stats()
            start_allocated_gpu_mem = torch.cuda.memory_allocated() / 1024**3

            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.time()
                end_reserved_gpu_mem = torch.cuda.memory_reserved() / 1024**3
                end_unavailable_cpu_mem_pct = psutil.virtual_memory().percent
                peak_allocated_gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
                end_allocated_gpu_mem = torch.cuda.memory_allocated() / (1024**3)

                elapsed_time = end_time - start_time
                reserved_mem_added = end_reserved_gpu_mem - start_reserved_gpu_mem
                cpu_mem_added = end_unavailable_cpu_mem_pct - start_unavailable_cpu_mem_pct
                peak_allocated_gpu_mem_delta = peak_allocated_gpu_mem - start_allocated_gpu_mem
                final_allocated_gpu_mem_delta = end_allocated_gpu_mem - start_allocated_gpu_mem

                cls.stats[func_name]["calls"] += 1
                cls.stats[func_name]["time"] += elapsed_time
                cls.stats[func_name]["gpu_mem_res"] += reserved_mem_added   
                cls.stats[func_name]["gpu_mem_peak"] += peak_allocated_gpu_mem
                cls.stats[func_name]["gpu_mem_peak_delta"] += peak_allocated_gpu_mem_delta
                cls.stats[func_name]["gpu_mem_final_delta"] += final_allocated_gpu_mem_delta
                cls.stats[func_name]["cpu_mem"] += cpu_mem_added

                print(f"[PROFILE] func {func_name}(...)"
                      f"    execution time: {elapsed_time:.4f}s\n"
                      f"    abs peak gpu mem: {peak_allocated_gpu_mem:.2f}GiB\n"
                      f"    peak gpu mem increase: {peak_allocated_gpu_mem_delta:.2f}GiB\n"
                      f"    persistent gpu mem increase: {final_allocated_gpu_mem_delta:.2f}GiB\n"
                    #   f"    gpu mem reserved: {reserved_mem_added:.2f}GiB"
                      )

                if final_allocated_gpu_mem_delta > 0.01: # ~10 MB
                    print(f"        ⚠️ Possible Leak!\n")
                
        return wrapper
    
    @classmethod
    def print_stats(self):
        print(f"\n\n=========Profiler Stats=========")

        for func, stats in self.stats.items():
            avg_time = stats["time"] / stats["calls"] if stats["calls"] else 0.0
            avg_gpu_mem = stats["gpu_mem_res"] / stats["calls"] if stats["calls"] else 0.0
            avg_gpu_mem_peak = stats["gpu_mem_peak"] / stats["calls"] if stats["calls"] else 0.0
            avg_gpu_mem_peak_delta = stats["gpu_mem_peak_delta"] / stats["calls"] if stats["calls"] else 0.0
            avg_gpu_mem_final = stats["gpu_mem_final_delta"] / stats["calls"] if stats["calls"] else 0.0
            avg_cpu_mem = stats["cpu_mem"] / stats["calls"] if stats["calls"] else 0.0
            
            print(
                f"{func}: calls={stats['calls']}    "
                f"total_time={stats['time']:.4f}s avg_time={avg_time:.4f}s    "
                f"avg_gpu_mem={avg_gpu_mem:.3f}GiB    "
                f"avg_gpu_mem_peak={avg_gpu_mem_peak:.3f}GiB    "
                f"avg_gpu_mem_peak_delta={avg_gpu_mem_peak_delta:.3f}GiB    "
                f"avg_gpu_mem_final_delta={avg_gpu_mem_final:.3f}GiB    "
                f"avg_cpu_mem={avg_cpu_mem:.3f}%    "
            )
        print(f"===============================\n\n")

# Create alias so `@profile decorator works `
profile = Profiler.profile 
atexit.register(Profiler.print_stats)