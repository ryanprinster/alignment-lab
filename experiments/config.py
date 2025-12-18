import pdb
class ConfigBase:
    def __init__(self):
        raise NotImplementedError("Use a concrete config subclass")
    
    def compile(self):
        pass

class SFTConfigBase(ConfigBase):
    def __init__(self):
        raise NotImplementedError("Use a concrete config subclass")

class RMConfigBase(ConfigBase):
    def __init__(self):
        raise NotImplementedError("Use a concrete config subclass")

class PPOConfigBase(ConfigBase):
    def __init__(self):
        raise NotImplementedError("Use a concrete config subclass")

class PPOConfig(PPOConfigBase):
    def __init__(self):
        self.eps = 0.2 
        self.N = 32 # parallel actors
        self.K = 10 # epochs
        self.M = 64 # minibatch
        self.num_train_iter = 75 
        self.beta = 0.01 # entropy regularization

        self.gamma = 0.99 # GAE gamma
        self.alpha = 3e-4 # lr
        self.lam = 0.95 # GAE lambda

        self.gymnasium_env_name = 'CartPole-v1'
        self.max_env_steps = 1e6 # Original paper benchmarks at 1 million steps
        self.video_folder_name = 'cartpole-ppo-videos'



class SFTConfig1(SFTConfigBase):
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


class RLFHCaseStudyConfig(SFTConfigBase, RMConfigBase):
    def __init__(self):
        # Based on https://arxiv.org/pdf/2403.17031

        # Detail 9 (SFT Training -> Setups) 
        self.num_epochs = 1 # (or 116,722 episodes)
        # Adam W Optimizer
        self.eps = 1e-5
        self.lr = 3e-6
        self.lr_final_ratio = 0.1
        
        self.batch_size = 32
        self.accumulation_steps = 1

        self.generation_temperature = 0.7

        # Checkpointing
        self.save_freq_steps = 100 * self.accumulation_steps
        self.save_interval_min = 60
        self.sft_model_path = "checkpoints/sft_final_checkpoint.pt"
        self.rm_model_path = "checkpoints/rm_final_checkpoint_v2.pt"
        
        self.keep_last_n=2
        self.temperature_scale_logits = False

        # Logging
        # self.log_weights_freq=None
        self.log_scalars_freq=self.accumulation_steps
        self.log_file_name="sft_training_log"

        # Efficiency
        self.enable_gradient_checkpointing = False
        self.enable_mixed_precision_training = True

        # RM 
        self.calculated_sft_bias = 4.6745805740356


        # Detail 7 (Disable dropout) aka there is no dropout
        # Detail 8 (Tech stack) Differences from the paper thus far:
        # --> Smaller model
        # --> Not using accelerate
        # --> No mixed precision training (yet)
        # --> No ZeRO Stage 2 (yet)
        # --> No ZeRO Stage 2 (yet)
        # --> Initial plan to use 1xH100 with gradient accumulation to trade training time for
        #     Memory and implementation time of ZeRO
    
    def compile(self):
        self._virtual_batch_size = self.batch_size * self.accumulation_steps

class RLFHPPOConfig(PPOConfigBase):
    def __init__(self):
        # Detail 20 (PPO Training -> Setups) 
        # Closely follows Stiennon et al. (2020) with modified learning rate
        # See Table 7 of https://arxiv.org/pdf/2403.17031


        # Detail 21 (PPO Training -> Re-use the SFT dataset and shuffle when reaches the end)
        # Note: PPO trains for 8.56 epochs relative to SFT. However, the 1B model
        # becomes over-optimized (7.1 point 3). 
        # Additionally, 8.56 epochs creates a training scale factor of 8.56x time and $$.
        # For these reasons, we train for 1 epoch.
        self.num_epochs = 9 # (1 epoch = 116,722 episodes)
        self.max_episodes = 4e5

        # Adam W Optimizer
        self.eps_adam = 1e-5
        self.lr = self.alpha = 3e-6
        self.lr_final_ratio = 0 # Paper does not state for PPO

        self.batch_size = 2 # Number of trajectories generated at a time

        self.mini_batch_accumulation_steps = 2

        self.beta = 0.05 # KL Penalty Coefficient for RLHF
        self.gamma = 1.0 # Discount factor
        self.lam = 0.95 # GAE
        
        self.N = self.num_mini_batches= 1 # N_mb = Number of mini-batches to process batch_size of trajectories
        self.K = 4 # PPO update per epoch
        self.M = 64 # minibatch

        self.eps_policy_clipping = 0.2
        self.eps_value_clipping = 0.2
        self.c1 = 0.1 # value func coeff
        self.clip_value_func_loss = True
        self.generation_temperature = 0.7 # Sampling temp
        self.temperature_scale_logits = True

        self.whiten_A = True
        self.whiten_rewards = True

        self.top_p_generation = 0.9

        self.hf_rm_model_name = "meta-llama/Llama-3.2-1B"
        self.hf_sft_model_name = "meta-llama/Llama-3.2-1B"
        self.hf_sft_model_revision = "main"
        self.hf_rm_model_revision = "main"


        # Checkpointing
        self.save_freq_steps = 3000
        self.keep_last_n = 2
        # self.save_interval_min = 60
        # self.load_checkpoint_path = "./checkpoints/checkpoint_best.pt"
        self.sft_model_path = "checkpoints/sft_final_checkpoint.pt"
        self.rm_model_path = "checkpoints/rm_final_checkpoint_v2.pt"
        self.calculated_sft_bias = 4.6745805740356
        self.resume_from_checkpoint = False
        self.policy_checkpoint_path = "checkpoints/policy__checkpoint_step_600.pt"
        self.value_checkpoint_path = "checkpoints/value__checkpoint_step_600.pt"


        # Logging
        # self.log_weights_freq=None
        self.log_samples_freq=10
        self.log_scalars_freq=self.mini_batch_accumulation_steps
        self.log_file_name="sft_training_log"

        # Efficiency
        self.enable_gradient_checkpointing = True
        self.enable_mixed_precision_training = True
        self.pre_compute_rm_scores = True

        self.compile()

    def compile(self):
        """
        Computes derived values and re-checks assertions
        """

        assert(self.batch_size % self.num_mini_batches == 0)
        self._mini_batch_size = int(self.batch_size / self.num_mini_batches)

        assert(self._mini_batch_size % self.mini_batch_accumulation_steps == 0)
        self._virtual_mini_batch_size = int(self._mini_batch_size / self.mini_batch_accumulation_steps) # Batch size that is actually getting trained, and hence is in memory



class CartPoleConfig():
    def __init__(self):
        pass

class ValueModelConfig():
    def __init__(self):
        pass