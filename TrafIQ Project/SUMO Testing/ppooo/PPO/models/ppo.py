import torch
import torch.nn as nn
import torch.optim as optim
from models.network import PolicyNN, ValueNet

class PPOAgent:
    def __init__(self, obs_dim, act_dim, cfg, device='cpu'):
        self.device = device
        self.policy = PolicyNN(obs_dim, act_dim, t_min=cfg.get("t_min",10), t_max=cfg.get("t_max",60)).to(device)
        self.value = ValueNet(obs_dim).to(device)

        self.pi_opt = optim.Adam(self.policy.parameters(), lr=cfg.get("lr",3e-4))
        self.vf_opt = optim.Adam(self.value.parameters(), lr=cfg.get("lr",3e-4))

        self.clip_ratio = cfg.get("clip_ratio", 0.2)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.value_coef = cfg.get("value_coef", 0.5)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)
        self.update_epochs = cfg.get("update_epochs", 10)
        self.minibatch_size = cfg.get("minibatch_size", 64)

    @torch.no_grad()
    def act(self, obs):
        act, logp = self.policy.sample(obs)
        v = self.value(obs if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32, device=self.device))
        return act.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    def update(self, data):
        obs = data['obs']
        act = data['act']
        ret = data['ret']
        adv = data['adv']
        logp_old = data['logp']

        N = obs.shape[0]
        idxs = torch.randperm(N, device=self.device)
        for _ in range(self.update_epochs):
            for start in range(0, N, self.minibatch_size):
                b = idxs[start:start+self.minibatch_size]
                b_obs = obs[b]
                b_act = act[b]
                b_ret = ret[b]
                b_adv = adv[b]
                b_logp_old = logp_old[b]

                new_logp = self.policy.log_prob(b_obs, b_act)
                ratio = torch.exp(new_logp - b_logp_old)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * b_adv
                pi_loss = -torch.min(surr1, surr2).mean()

                mu, std = self.policy(b_obs)
                entropy = (0.5 + 0.5 * torch.log(2 * torch.pi * std.pow(2))).sum(-1).mean()

                v = self.value(b_obs)
                v_loss = nn.functional.mse_loss(v, b_ret)

                loss = pi_loss - self.entropy_coef * entropy + self.value_coef * v_loss

                self.pi_opt.zero_grad()
                self.vf_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), self.max_grad_norm)
                self.pi_opt.step()
                self.vf_opt.step()
