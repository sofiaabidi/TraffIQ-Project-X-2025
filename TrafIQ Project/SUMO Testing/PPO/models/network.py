# networks.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class PolicyNN(nn.Module):
    def __init__(self, obs_dim, act_dim, t_min=10.0, t_max=60.0):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_head = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            device = next(self.parameters()).device
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std).clamp(1e-6, 1e6)
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        dist = Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        logp_z = dist.log_prob(z).sum(-1)
        log_det = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        logp_a = logp_z - log_det
        act = 0.5 * (a + 1.0) * (self.t_max - self.t_min) + self.t_min
        return act, logp_a

    def log_prob(self, obs, act):
        # map action back to [-1,1]
        if isinstance(act, np.ndarray):
            device = next(self.parameters()).device
            act = torch.tensor(act, dtype=torch.float32, device=device)
        a = 2.0 * (act - self.t_min) / (self.t_max - self.t_min) - 1.0
        a = torch.clamp(a, -0.999999, 0.999999)
        z = 0.5 * torch.log((1 + a) / (1 - a))
        mu, std = self(obs)
        dist = Normal(mu, std)
        logp_z = dist.log_prob(z).sum(-1)
        log_det = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return logp_z - log_det

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_head = nn.Linear(64, 1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            device = next(self.parameters()).device
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.v_head(x).squeeze(-1)
    