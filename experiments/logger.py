from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
import atexit
import signal
import torch
import sys
import psutil

from experiments.profiler import profile


class Logger:
    def __init__(self, config):
        # Thanks to Claude here:
        self._closed = False
        atexit.register(self.close)  # Normal exit
        signal.signal(signal.SIGTERM, self._cleanup_signal)  # Pod termination
        signal.signal(signal.SIGINT, self._cleanup_signal)  # Ctrl+C

        self.config = config
        self.writer = SummaryWriter()
        self.best_loss = float("inf")
        self.init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ### Cleanup handling

    def __del__(self):
        self.close()

    def _cleanup_signal(self, signum, frame):  # Thanks to Claude here
        print(f"Received signal {signum}, cleaning up...")
        self.close()
        sys.exit(0)

    def close(self):
        if not self._closed and hasattr(self, "writer"):
            self.writer.close()
            self._closed = True
            print("TensorBoard writer closed")

    ### Logging Entrypoint

    def log(self, scalars, models, samples=None, log_file_name=None):
        if not self._closed:
            self.log_to_tensorboard(models, scalars)
            self.log_to_terminal(scalars, samples)
            self.log_to_file(scalars, log_file_name)

    ### Console/Terminal Logging

    def log_to_terminal(self, scalars, samples):
        # Log scalars
        log_str = "\n\n\n"
        for key in scalars.keys():
            log_str += f"{key}: {scalars[key]}, "

        mem_info_str = self._get_memory_usage_info()
        log_str += f"{mem_info_str}\n\n\n"
        print(log_str)

        # Log samples
        if hasattr(self.config, "log_samples_freq") and samples is not None:
            if (
                scalars["global_step"] % self.config.log_samples_freq == 0
                and scalars["k"] == 0
            ):
                print(f"SAMPLES AT {scalars['global_step']} STEPS")
                for k, v in samples.items():
                    print(f"\n{k}: {v}\n")

    def _get_memory_usage_info(self):
        mem_usage_info_str = ""
        if torch.cuda.is_available():
            # for my small brain: recall that / 1024**3 is Bytes --> GB
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_mem_pct_allocated = (gpu_mem_used / gpu_mem_total) * 100
            gpu_mem_pct_reserved = (gpu_mem_reserved / gpu_mem_total) * 100
            gpu_info = f"GPU Allocated: {gpu_mem_used:.1f}/{gpu_mem_total:.1f}GB ({gpu_mem_pct_allocated:.1f}%) "
            gpu_info += f"GPU Reserved: {gpu_mem_reserved:.1f}/{gpu_mem_total:.1f}GB ({gpu_mem_pct_reserved:.1f}%)"
            mem_usage_info_str = gpu_info
        else:
            gpu_info = "CPU"
            cpu_mem = psutil.virtual_memory()
            cpu_mem_used = cpu_mem.used / 1024**3
            cpu_mem_total = cpu_mem.total / 1024**3
            cpu_mem_pct = 100.0 * cpu_mem_used / cpu_mem_total
            cpu_mem_pct_real = cpu_mem.percent
            cpu_mem_used_real = cpu_mem_pct_real * cpu_mem_total / 100
            cpu_info = f"CPU Mem Used: {cpu_mem_used:.1f}/{cpu_mem_total:.1f}GB ({cpu_mem_pct:.1f}%) "
            cpu_info += f"CPU Mem Unavailable: {cpu_mem_used_real:.1f}/{cpu_mem_total:.1f}GB ({cpu_mem_pct_real:.1f}%) "
            mem_usage_info_str = cpu_info

        return mem_usage_info_str

    ### Tensorboard Logging

    def log_to_tensorboard(self, models, scalars):
        if not hasattr(self, "writer"):
            print("No valid Tensorboard writer. Skipping tensorboard logging.")
            return

        if not "global_step" in scalars.keys():
            print("No global_step scalar. Skipping tensorboard logging.")
            return

        if hasattr(self.config, "log_weights_freq"):
            if scalars["global_step"] % self.config.log_weights_freq == 0:
                self._log_weights_and_grads_to_tensorboard(
                    models, scalars["global_step"]
                )
        if hasattr(self.config, "log_scalars_freq"):
            if scalars["global_step"] % self.config.log_scalars_freq == 0:
                self._log_scalars_to_tensorboard(scalars, scalars["global_step"])

        self.writer.flush()

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
                    self.writer.add_histogram(
                        f"{model_name}{name}.grad", param.grad, global_step=global_step
                    )
                if param.data is not None:
                    self.writer.add_histogram(
                        f"{model_name}{name}.weight",
                        param.data,
                        global_step=global_step,
                    )

    @profile
    def _log_scalars_to_tensorboard(self, scalars, global_step):
        for name in scalars.keys():
            self.writer.add_scalar(name, scalars[name], global_step)

    ### File Logging

    def log_to_file(self, scalars, log_file_name=None):
        log_data = {}
        for key in scalars.keys():
            log_data[key] = scalars[key]
            log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        log_file_name = (
            log_file_name or getattr(self.config, "log_file_name", None) or "log_at"
        )

        with open(f"{log_file_name}_{self.init_time}.jsonl", "a") as f:
            f.write(json.dumps(log_data) + "\n")
