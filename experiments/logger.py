from torch.utils.tensorboard import SummaryWriter
import atexit
import signal
import torch
import sys

from experiments.profiler import profile

class Logger():
    def __init__(self,log_weights_freq=100,log_scalars_freq=5):
        self.writer = SummaryWriter()
        self.best_loss = float('inf')
        self.log_weights_freq = log_weights_freq
        self.log_scalars_freq = log_scalars_freq

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

       
    def log(self, scalars, models, epoch, global_step):
        if not self._closed and hasattr(self, 'writer'):
            if global_step % self.log_weights_freq == 0: 
                self._log_weights_and_grads(models, global_step)
            if global_step % self.log_scalars_freq == 0: 
                self._log_scalars(scalars, global_step)
                
            print("epoch: ", epoch, "global_step: ", global_step, " loss: ", scalars["loss"], "\n\n\n")

            self.writer.flush()

    @profile 
    def _log_weights_and_grads(self, models, global_step):
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
    def _log_scalars(self, scalars, global_step):
        for name in scalars.keys():
            self.writer.add_scalar(name, scalars[name], global_step)