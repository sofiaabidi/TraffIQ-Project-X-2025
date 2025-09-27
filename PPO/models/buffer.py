import numpy as np
import torch

class RolloutBuffer:
    def _init_(self, num_agents, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device='cpu'):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.device = device
        
        self.obs = np.zeros((size, num_agents, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, num_agents, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, num_agents), dtype=np.float32)
        self.val = np.zeros((size, num_agents), dtype=np.float32)
        self.logp = np.zeros((size, num_agents), dtype=np.float32)
        
        self.adv = np.zeros((size, num_agents), dtype=np.float32)
        self.ret = np.zeros((size, num_agents), dtype=np.float32)
        self.ptr = 0

    def store(self, obs_list, act_list, rew_list, val_list, logp_list):
        for i in range(self.num_agents):
            self.obs[self.ptr, i] = obs_list[i]
            self.act[self.ptr, i] = act_list[i]
            self.rew[self.ptr, i] = rew_list[i]
            self.val[self.ptr, i] = val_list[i]
            self.logp[self.ptr, i] = logp_list[i]
        
        self.ptr += 1

    def finish_path(self, last_vals=None):
        if last_vals is None:
            last_vals = np.zeros(self.num_agents, dtype=np.float32)
        
        for i in range(self.num_agents):
            rews = np.append(self.rew[:self.ptr, i], last_vals[i])
            vals = np.append(self.val[:self.ptr, i], last_vals[i])
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            gae = 0.0
            for t in reversed(range(self.ptr)):
                gae = deltas[t] + self.gamma * self.lam * gae
                self.adv[t, i] = gae
            self.ret[:self.ptr, i] = self.adv[:self.ptr, i] + self.val[:self.ptr, i]

    def get(self, agent_id):
        obs_agent = self.obs[:self.ptr, agent_id]
        act_agent = self.act[:self.ptr, agent_id]
        ret_agent = self.ret[:self.ptr, agent_id]
        logp_agent = self.logp[:self.ptr, agent_id]
        adv_agent = self.adv[:self.ptr, agent_id]

        adv_agent = (adv_agent - adv_agent.mean()) / (adv_agent.std() + 1e-8)
        data = {
            'obs': torch.tensor(obs_agent, dtype=torch.float32, device=self.device),
            'act': torch.tensor(act_agent, dtype=torch.float32, device=self.device),
            'ret': torch.tensor(ret_agent, dtype=torch.float32, device=self.device),
            'logp': torch.tensor(logp_agent, dtype=torch.float32, device=self.device),
            'adv': torch.tensor(adv_agent, dtype=torch.float32, device=self.device)
        }
        return data
    
    def get_all_agents(self):
        data = []
        for i in range(self.num_agents):
            agent_data = self.get(i)
            data.append(agent_data)
        return data

    def reset(self):
        self.ptr = 0