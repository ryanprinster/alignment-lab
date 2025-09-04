from datetime import time
import json
from torch.utils.tensorboard import SummaryWriter
import atexit
import signal
import torch
import sys
import psutil

from experiments.profiler import profile

class Logger():
    def __init__(self, config):

        self.config = config
        self.writer = SummaryWriter()
        self.best_loss = float('inf')

        self._closed = False
        # Thanks to Claude here:
        atexit.register(self.close)  # Normal exit
        signal.signal(signal.SIGTERM, self._cleanup_signal)  # Pod termination
        signal.signal(signal.SIGINT, self._cleanup_signal)   # Ctrl+C
    
    def __del__(self):
        self.close()

    # Thanks to Claude here:
    def _cleanup_signal(self, signum, frame):
        print(f"Received signal {signum}, cleaning up...")
        self.close()
        sys.exit(0)

    def close(self):
        if not self._closed and hasattr(self, 'writer'):
            self.writer.close()
            self._closed = True
            print("TensorBoard writer closed")

       
    def log(self, scalars, models, epoch, global_step, lr):
        if not self._closed and hasattr(self, 'writer'):
            if hasattr(self.config, 'log_weights_freq'):
                if global_step % self.config.log_weights_freq == 0: 
                    self._log_weights_and_grads_to_tensorboard(models, global_step)
            if hasattr(self.config, 'log_scalars_freq'):
                if global_step % self.config.log_scalars_freq == 0: 
                    self._log_scalars_to_tensorboard(scalars, global_step)
                
            mem_info_str = self._get_memory_usage_info()
            print(f"\n\n\n epoch: {epoch}, global_step: {global_step}, loss: {scalars['loss']}, {mem_info_str}\n\n\n")

            self.writer.flush()

    def _get_memory_usage_info(self):
        mem_usage_info_str = ""
        if torch.cuda.is_available():
            # / 1024**3 is Bytes --> GB
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3 
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 
            gpu_mem_pct_allocated = (gpu_mem_used / gpu_mem_total) * 100
            gpu_mem_pct_reserved = (gpu_mem_used / gpu_mem_total) * 100
            gpu_info = f"GPU Allocated: {gpu_mem_used:.1f}/{gpu_mem_total:.1f}GB ({gpu_mem_pct_allocated:.1f}%) "
            gpu_info += f"GPU Reserved: {gpu_mem_reserved:.1f}/{gpu_mem_total:.1f}GB ({gpu_mem_pct_reserved:.1f}%)"
            mem_usage_info_str = gpu_info
        else:
            gpu_info = "CPU"
            cpu_mem = psutil.virtual_memory()
            cpu_mem_used = cpu_mem.used / 1024**3
            cpu_mem_total = cpu_mem.total / 1024**3 
            cpu_mem_pct = 100.0 * cpu_mem_used/cpu_mem_total
            cpu_mem_pct_real = cpu_mem.percent
            cpu_mem_used_real = cpu_mem_pct_real * cpu_mem_total / 100
            cpu_info = f"CPU Mem Used: {cpu_mem_used:.1f}/{cpu_mem_total:.1f}GB ({cpu_mem_pct:.1f}%) "
            cpu_info += f"CPU Mem Unavailable: {cpu_mem_used_real:.1f}/{cpu_mem_total:.1f}GB ({cpu_mem_pct_real:.1f}%) "
            mem_usage_info_str = cpu_info

        return mem_usage_info_str

    
    @profile
    def _write_step_data(self, step, loss, lr):
        log_data = {"step": step, "loss": loss, "lr": lr, "timestamp": time.time()}
        with open(self.config.log_file_name + ".jsonl", "a") as f:
            f.write(json.dumps(log_data) + "\n")

    @profile 
    def _log_weights_and_grads_to_tensorboard(self, models, global_step):
        for model in models:
            model_name = model.__class__.__name__
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if param.grad.numel() == 0:
                        print("Gradients are zero and technically empty")
                    if torch.isinf(param.grad).any():
                        print("Gradients are inf")
                    self.writer.add_histogram(f'{model_name}{name}.grad', param.grad, global_step=global_step)
                if param.data is not None:
                    self.writer.add_histogram(f'{model_name}{name}.weight', param.data, global_step=global_step)
    @profile
    def _log_scalars_to_tensorboard(self, scalars, global_step):
        for name in scalars.keys():
            self.writer.add_scalar(name, scalars[name], global_step)