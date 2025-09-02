import os
from datetime import datetime
import torch
import time

from experiments.profiler import profile

class Checkpointer:
    def __init__(self, checkpoint_dir="./checkpoints", keep_last_n=3, save_freq_steps=500, save_freq_epochs=1, save_interval_min=60):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.best_loss = float('inf')
        self.save_freq_steps = save_freq_steps
        self.save_freq_epochs = save_freq_epochs
        self.save_interval_secs = save_interval_min * 60  
        self.last_save_time = time.time()

        os.makedirs(checkpoint_dir, exist_ok=True)

    @profile
    def save_checkpoint(self, model, optimizer, global_step, epoch, loss):        
        
        path = self._should_save_checkpoint(global_step, epoch, loss)
        if path:
            checkpoint = self._build_checkpoint(model, optimizer, global_step, epoch, loss)
            torch.save(checkpoint, path)
            self._cleanup_old_checkpoints()

    def load_checkpoint(self):
        # [TODO] implement
        pass

    def _should_save_checkpoint(self, global_step, epoch, loss):
        path = None
        
        if loss < self.best_loss:
            self.best_loss = loss
            path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")

        elif global_step % self.save_freq_steps == 0:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
        
        elif time.time() - self.last_save_time >= self.save_interval_secs:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_at_{datetime.now().isoformat()}.pt")
            self.last_save_time = time.time()
        
        return path


    def _build_checkpoint(self, model, optimizer, global_step, epoch, loss):
        return {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }

    def _cleanup_old_checkpoints(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                           if (f.startswith("checkpoint_step_") or f.startswith("checkpoint_at_"))]
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        for old_checkpoint in checkpoints[:-self.keep_last_n]:
            old_path = os.path.join(self.checkpoint_dir, old_checkpoint)
            os.remove(old_path)
            print(f"Removed old checkpoint: {old_checkpoint}")