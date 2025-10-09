# buffer.py
import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device='cpu'):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.start = 0
        self.max_size = size
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def store(self, obs, act, rew, val, logp, done):
        assert self.ptr < self.max_size, "Buffer overflow"
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.logp[self.ptr] = logp
        self.done[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        # compute GAE and returns for last trajectory slice
        path_slice = slice(self.start, self.ptr)
        rews = np.append(self.rew[path_slice], last_val)
        vals = np.append(self.val[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = discount_cumsum(deltas, self.gamma * self.lam)
        ret = adv + vals[:-1]

        if not hasattr(self, 'adv'):
            self.adv = np.zeros(self.max_size, dtype=np.float32)
            self.ret = np.zeros(self.max_size, dtype=np.float32)

        self.adv[path_slice] = adv
        self.ret[path_slice] = ret
        self.start = self.ptr

    def get(self):
        assert self.ptr == self.max_size, "Buffer must be full before get()"
        adv = self.adv.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = dict(
            obs=torch.as_tensor(self.obs, dtype=torch.float32, device=self.device),
            act=torch.as_tensor(self.act, dtype=torch.float32, device=self.device),
            ret=torch.as_tensor(self.ret, dtype=torch.float32, device=self.device),
            logp=torch.as_tensor(self.logp, dtype=torch.float32, device=self.device),
            adv=torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        )
        # reset
        self.ptr = 0
        self.start = 0
        return data

def discount_cumsum(x, discount):
    y = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        y[t] = running
    return y
