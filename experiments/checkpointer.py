import os
from datetime import datetime
import torch
import time

from experiments.profiler import profile

class Checkpointer:
    def __init__(self, config, checkpoint_dir="./checkpoints", keep_last_n=2, save_freq_steps=200, save_freq_epochs=1, save_interval_min=60):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = self.config.keep_last_n or keep_last_n
        self.best_loss = float('inf')
        self.save_freq_steps = save_freq_steps
        self.save_freq_epochs = save_freq_epochs
        self.save_interval_secs = save_interval_min * 60  
        self.last_save_time = time.time()

        os.makedirs(checkpoint_dir, exist_ok=True)

    @profile
    def save_checkpoint(self, model, optimizer, global_step, epoch, loss, checkpoint_prefix, final_checkpoint=False):        
        
        should_save_checkpoint, path = self._should_save_checkpoint(global_step, loss, checkpoint_prefix, final_checkpoint)
        
        if should_save_checkpoint:
            self._save_checkpoint(path, model, optimizer, global_step, epoch, loss)

    @profile
    def load_model(self, checkpoint_path, model, device):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def _should_save_checkpoint(self, global_step, loss, checkpoint_prefix, final_checkpoint):
        should_save_checkpoint = False
        path = None

        if final_checkpoint:
            path = os.path.join(self.checkpoint_dir, f"{checkpoint_prefix}_final_checkpoint.pt")
            return True, path

        if hasattr(self.config, 'save_freq_steps'):
            self.save_freq_steps = self.config.save_freq_steps
        if hasattr(self.config, 'save_interval_min'):
            self.save_interval_secs = self.config.save_interval_min * 60
        
        # TODO: Build in more robust/smarter best loss/reward based and time based checkpointing

        # Don't save so frequently at the beginning, slowing things down
        # if (loss < self.best_loss) and (global_step > self.save_freq_steps):
        #     self.best_loss = loss
        #     path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
        #     should_save_checkpoint = True

        if global_step % self.save_freq_steps == 0 and global_step != 0:
            path = os.path.join(self.checkpoint_dir, f"{checkpoint_prefix}_checkpoint_step_{global_step}.pt")
            should_save_checkpoint = True
        
        # elif time.time() - self.last_save_time >= self.save_interval_secs:
        #     path = os.path.join(self.checkpoint_dir, f"checkpoint_at_{datetime.now().isoformat()}.pt")
        #     self.last_save_time = time.time()
        #     should_save_checkpoint = True
        
        return should_save_checkpoint, path

    @profile
    def _save_checkpoint(self, path, model, optimizer, global_step, epoch, loss):
        checkpoint = self._build_checkpoint(model, optimizer, global_step, epoch, loss)
        torch.save(checkpoint, path)
        self._cleanup_old_checkpoints()

    def _build_checkpoint(self, model, optimizer, global_step, epoch, loss):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        return checkpoint

    def _cleanup_old_checkpoints(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                        if (f.__contains__("checkpoint_step_") or f.__contains__("checkpoint_at_"))]
        
        def get_step_number(filename):
            # policy__checkpoint_step_123.pt or policy__checkpoint_at_123.pt
            parts = filename.split('_')
            # Find 'step' or 'at' and get the next part
            for i, part in enumerate(parts):
                if part in ['step', 'at'] and i + 1 < len(parts):
                    return int(parts[i + 1].split('.')[0])
            return 0
        
        checkpoints.sort(key=get_step_number)
        
        for old_checkpoint in checkpoints[:-self.keep_last_n]:
            old_path = os.path.join(self.checkpoint_dir, old_checkpoint)
            os.remove(old_path)
            print(f"Removed old checkpoint: {old_checkpoint}")

    @profile
    def load_checkpoint(self, checkpoint_path, model, device, optimizer=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded model state from step {checkpoint.get('global_step', 'unknown')}")
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
        
        # Return training state for resuming
        training_state = {
            'global_step': checkpoint.get('global_step', 0),
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
        
        print(f"Checkpoint loaded - Step: {training_state['global_step']}, "
            f"Epoch: {training_state['epoch']}, Loss: {training_state['loss']:.4f}")
        
        return training_state
