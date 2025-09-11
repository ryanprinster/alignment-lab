import os
from datetime import datetime
import torch
import time

from experiments.profiler import profile

class Checkpointer:
    def __init__(self, config, checkpoint_dir="./checkpoints", keep_last_n=2, save_freq_steps=200, save_freq_epochs=1, save_interval_min=60):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.best_loss = float('inf')
        self.save_freq_steps = save_freq_steps
        self.save_freq_epochs = save_freq_epochs
        self.save_interval_secs = save_interval_min * 60  
        self.last_save_time = time.time()

        os.makedirs(checkpoint_dir, exist_ok=True)

    @profile
    def save_checkpoint(self, model, optimizer, global_step, epoch, loss, final_checkpoint=False):        
        
        should_save_checkpoint, path = self._should_save_checkpoint(global_step, loss, final_checkpoint)
        
        if should_save_checkpoint:
            self._save_checkpoint(path, model, optimizer, global_step, epoch, loss)

    @profile
    def load_model(self, checkpoint_path, model, device):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    @profile
    def load_checkpoint(self, checkpoint_path, model, device, optimizer=None):
        pass

    def _should_save_checkpoint(self, global_step, loss, final_checkpoint):
        should_save_checkpoint = False
        path = None

        if final_checkpoint:
            path = os.path.join(self.checkpoint_dir, f"final_checkpoint.pt")
            return True, path

        if hasattr(self.config, 'save_freq_steps'):
            self.save_freq_steps = self.config.save_freq_steps
        if hasattr(self.config, 'save_interval_min'):
            self.save_interval_secs = self.config.save_interval_min * 60
        
        
        # Don't save so frequently at the beginning, slowing things down
        # if (loss < self.best_loss) and (global_step > self.save_freq_steps):
        #     self.best_loss = loss
        #     path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
        #     should_save_checkpoint = True

        if global_step % self.save_freq_steps == 0:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
            should_save_checkpoint = True
        elif time.time() - self.last_save_time >= self.save_interval_secs:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_at_{datetime.now().isoformat()}.pt")
            self.last_save_time = time.time()
            should_save_checkpoint = True
        
        return should_save_checkpoint, path

    @profile
    def _save_checkpoint(self, path, model, optimizer, global_step, epoch, loss):
        checkpoint = self._build_checkpoint(model, optimizer, global_step, epoch, loss)
        torch.save(checkpoint, path)
        self._cleanup_old_checkpoints()

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