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

class SFTConfig():
    def __init__(self):
        # TODO: Could move optimizer, lr scheduler, data to config

        # Values taken from https://arxiv.org/pdf/2203.02155 Section C
        self.beta_1 = 0.9
        self.beta_2 = 0.95
        self.num_epochs = 16
        self.dropout = 0.2
        # self.lr = 9.65e-6
        self.final_lr = 0.1
        self.batch_size = 2 # 32
        self.test_pct = 0.1
        # self.eps = 1e-4

        # Values taken from https://arxiv.org/pdf/2403.17031 Table 3
        # self.num_epochs = 1 # (or 116,722 episodes)
        self.eps = 1e-5
        self.lr = 3e-6
        # self.batch_size = 128
        # Adam W Optimizer
        # TL;DR dataset

class CartPoleConfig():
    def __init__(self):
        pass

class ValueModelConfig():
    def __init__(self):
        pass