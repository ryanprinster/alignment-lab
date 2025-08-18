from functools import reduce
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter



class MLPValue(nn.Module):
    def __init__(self, obs_dim, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]

        self.l1 = nn.Linear(obs_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1],1)

    def forward(self, x):
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(dim=0)
        
        x = self.l1(x)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)

        if single_input:
            x = x.squeeze(dim=0)

        return x

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]

        self.l1 = nn.Linear(obs_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, x):
        # TODO: Make this more clean
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(dim=0)
        
        x = self.l1(x)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)
        # Could do log probs but this works fine
        x = torch.softmax(x, dim=-1)
        if single_input:
            x = x.squeeze(dim=0)
        return x
