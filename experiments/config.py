class PPOConfig():
    def __init__(self):
        self.eps = 0.2 
        self.N = 32 # parallel actors
        self.K = 10 # epochs
        self.M = 64 # minibatch
        self.num_train_iter = 75 
        self.beta = 0.01

        self.gamma = 0.99
        self.alpha = 3e-4 # lr
        self.lam = 0.95

class SFTConfig1():
    def __init__(self):
        # Based on https://arxiv.org/pdf/2203.02155 (See Section C)

        # Adam W Optimizer
        self.beta_1 = 0.9
        self.beta_2 = 0.95
        self.num_epochs = 16
        self.dropout = 0.2
        self.lr = 9.65e-6
        self.final_lr = 0.1
        self.batch_size = 2 # 32
        self.test_pct = 0.1
        # self.eps = 1e-4

      

class SFTConfig2():
    def __init__(self):
        # Based on https://arxiv.org/pdf/2403.17031

        # Detail 9 (SFT Training -> Setups) 
        self.num_epochs = 1 # (or 116,722 episodes)
        # Adam W Optimizer
        self.eps = 1e-5
        self.lr = 3e-6
        self.lr_final_ratio = 0.1
        # self.batch_size = 32
        # self.virtual_batch_size = 128
        self.batch_size = 2
        self.accumulation_steps = 4
        self._virtual_batch_size = self.batch_size * self.accumulation_steps
        self.generation_temperature = 0.7

        # Checkpointing
        self.save_freq_steps = 100 * self.accumulation_steps
        self.save_interval_min = 60
        self.load_checkpoint_path = "./checkpoints/checkpoint_best.pt"

        # Logging
        # self.log_weights_freq=None
        self.log_scalars_freq=self.accumulation_steps
        self.log_file_name="sft_training_log"

        # Efficiency
        self.enable_gradient_checkpointing = False
        self.enable_mixed_precision_training = True

        # TODO: Could move optimizer, lr scheduler, data to config

        # Detail 7 (Disable dropout) aka there is no dropout
        # Detail 8 (Tech stack) Differences from the paper thus far:
        # --> Smaller model
        # --> Not using accelerate
        # --> No mixed precision training (yet)
        # --> No ZeRO Stage 2 (yet)
        # --> No ZeRO Stage 2 (yet)
        # --> Initial plan to use 1xH100 with gradient accumulation to trade training time for
        #     Memory and implementation time of ZeRO
        


class CartPoleConfig():
    def __init__(self):
        pass

class ValueModelConfig():
    def __init__(self):
        pass