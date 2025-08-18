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

class CartPoleConfig():
    def __init__(self):
        pass

class ValueModelConfig():
    def __init__(self):
        pass