try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('matplotlib', 'inline')
except Exception:
    pass

import os, sys, time, json, numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from collections import defaultdict, namedtuple
import random

#sumo config (check your paths and env variables)
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/home/jupyter-yashvi_mehta/sumo"
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

try:
    import traci
except ImportError:
    raise ImportError("TraCI not found. Make sure SUMO is installed and SUMO_HOME is set correctly.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('SUMO_HOME:', os.environ.get('SUMO_HOME'))

sumo_binary = "/home/jupyter-yashvi_mehta/.local/bin/sumo"
sumo_config = "/home/jupyter-yashvi_mehta/env/gid.sumocfg"
sumo_step_length = 1.0
MAX_EPISODE_STEPS = 1800

sumo_cmd_template = [
    sumo_binary, "-c", sumo_config,
    "--step-length", str(sumo_step_length),
    "--no-step-log", "true"
]

NUM_EPISODES = 12500
ROLLOUT_STEPS = 1024
PPO_EPOCHS = 10
MINI_BATCHES = 8
GAMMA = 0.95
GAE_LAMBDA = 0.995
CLIP_EPS = 0.2
POLICY_LR = 2e-5
VALUE_LR = 2e-5
ENTROPY_COEF = 0.0005
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

GREEN_OPTIONS = [15,20,25,30,35,40]
YELLOW_TIME = 3
MAX_QUEUE_NORM = 20.0

def discover_traffic_lights():
    tls_info = {}
    tls_ids = traci.trafficlight.getIDList()
    for tls in tls_ids:
        lanes = traci.trafficlight.getControlledLanes(tls)
        links = traci.trafficlight.getControlledLinks(tls)
        try:
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)
            phases = []
            for pr in program:
                for ph in pr.getPhases():
                    phases.append({'state': ph.getState(), 'duration': ph.getDuration()})
        except Exception:
            phases = []
        tls_info[tls] = {'controlled_lanes': lanes, 'controlled_links': links, 'phases': phases}
    return tls_info

class MultiAgentSumoEnv:
    def __init__(self, green_options=None, yellow_time=YELLOW_TIME, max_queue_norm=MAX_QUEUE_NORM):
        self.green_options = green_options if green_options is not None else GREEN_OPTIONS
        self.yellow_time = yellow_time
        self.max_queue_norm = max_queue_norm
        self.tls_info = None
        self.current_step = 0
        self.last_switch = {}
        self.phase_index = {}

    def start_sumo(self, sumo_cmd, label=None):
        if label is not None and traci.connection.has(label):
            traci.connection.get(label).close()
        else:
            try:
                traci.close()
            except Exception:
                pass
        traci.start(sumo_cmd, label=label)
        for _ in range(5):
            if traci.simulation.getMinExpectedNumber()>0:
                traci.simulationStep()
        self.tls_info = discover_traffic_lights()
        for tls in self.tls_info:
            self.phase_index[tls] = 0
            self.last_switch[tls] = 0
        self.current_step = 0
        return self.tls_info

    def close_sumo(self):
        try:
            traci.close()
        except Exception:
            pass

    def _lane_queue(self, lane_id):
        return traci.lane.getLastStepHaltingNumber(lane_id)

    def _lane_waiting(self, lane_id):
        vids = traci.lane.getLastStepVehicleIDs(lane_id)
        if not vids: return 0.0
        return sum(traci.vehicle.getWaitingTime(v) for v in vids)

    def get_local_obs(self, tls_id):
        lanes = self.tls_info[tls_id]['controlled_lanes']
        if len(lanes)==0:
            return np.zeros(8, dtype=np.float32)
        queues = [self._lane_queue(l) for l in lanes]
        waiting = [self._lane_waiting(l) for l in lanes]
        norm_queues = [min(q/self.max_queue_norm,1.0) for q in queues]
        norm_waiting = [min(w/300.0,1.0) for w in waiting]
        phase_idx = self.phase_index.get(tls_id,0)
        phase_encoding = [0.0]*4
        phase_encoding[phase_idx%4] = 1.0
        obs = np.array(norm_queues + norm_waiting + [sum(queues)/max(1,len(queues))] + phase_encoding, dtype=np.float32)
        return obs

    def get_global_state(self):
        parts = []
        for tls in sorted(self.tls_info.keys()):
            lanes = self.tls_info[tls]['controlled_lanes']
            queues = [self._lane_queue(l) for l in lanes]
            parts.extend([len(lanes), sum(queues)])
        return np.array(parts, dtype=np.float32)

    def compute_multi_objective_reward(self, tls_id, prev_metrics):
        lanes = self.tls_info[tls_id]['controlled_lanes']
        queues = sum(self._lane_queue(l) for l in lanes)
        vehicle_ids = traci.vehicle.getIDList()
        total_waiting = 0.0
        total_speed = 0.0
        total_co2 = 0.0
        total_fuel = 0.0
        for vid in vehicle_ids:
            total_waiting += traci.vehicle.getWaitingTime(vid)
            total_speed += traci.vehicle.getSpeed(vid)
            try:
                total_co2 += traci.vehicle.getCO2Emission(vid)
                total_fuel += traci.vehicle.getFuelConsumption(vid)
            except Exception:
                pass
        cur = {'queues':queues,'waiting':total_waiting,'speed':total_speed,'co2':total_co2,'fuel':total_fuel,'throughput':len(vehicle_ids)}
        if prev_metrics is None:
            return 0.0, cur
        waiting_imp = (prev_metrics['waiting'] - cur['waiting'])/10.0
        queue_imp = (prev_metrics['queues'] - cur['queues'])/100.0
        co2_imp = (prev_metrics['co2'] - cur['co2'])/1000.0
        fuel_imp = (prev_metrics['fuel'] - cur['fuel'])/1000.0
        throughput_imp = (cur['throughput'] - prev_metrics['throughput'])/100.0
        speed_imp = (cur['speed'] - prev_metrics['speed'])/10.0
        weights = {'waiting':-1.0,'queue':-0.8,'co2':-0.5,'fuel':-0.3,'throughput':1.0,'speed':0.8}
        reward = (weights['waiting']*waiting_imp + weights['queue']*queue_imp +
                  weights['co2']*(co2_imp/1000.0) + weights['fuel']*(fuel_imp/1000.0) +
                  weights['throughput']*throughput_imp + weights['speed']*speed_imp)
        if cur['queues']>15:
            reward -= (cur['queues']-15)*0.5
        return float(reward), cur

    def apply_action(self, tls_id, action_idx):
        duration = self.green_options[action_idx]
        target_phase = 0 if (action_idx % 2)==0 else 2
        if target_phase != self.phase_index.get(tls_id,0) and (self.current_step - self.last_switch[tls_id]) >= 5:
            yellow_phase = self.phase_index[tls_id] + 1
            try:
                traci.trafficlight.setPhase(tls_id, yellow_phase)
            except Exception:
                pass
            for _ in range(self.yellow_time):
                if traci.simulation.getMinExpectedNumber()>0:
                    traci.simulationStep(); self.current_step += 1
            self.phase_index[tls_id] = target_phase
            self.last_switch[tls_id] = self.current_step
        try:
            traci.trafficlight.setPhaseDuration(tls_id, duration)
        except Exception:
            pass
        for _ in range(duration):
            if traci.simulation.getMinExpectedNumber()>0:
                traci.simulationStep(); self.current_step += 1

    def step(self, actions_dict, prev_metrics_dict):
        for tls, act in actions_dict.items():
            self.apply_action(tls, act)
        obs = {}
        rewards = {}
        new_metrics = {}
        for tls in self.tls_info:
            obs[tls] = self.get_local_obs(tls)
            rew, cur = self.compute_multi_objective_reward(tls, prev_metrics_dict.get(tls))
            rewards[tls] = rew
            new_metrics[tls] = cur
        done = (traci.simulation.getMinExpectedNumber()==0) or (self.current_step>=MAX_EPISODE_STEPS)
        return obs, rewards, done, new_metrics, {}

    def reset_env_state(self):
        self.current_step = 0
        for tls in self.tls_info:
            self.phase_index[tls] = 0
            self.last_switch[tls] = 0

class ActorNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden=[128,128]):
        super().__init__()
        layers=[]
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev,h)); layers.append(nn.ReLU()); prev = h
        self.body = nn.Sequential(*layers)
        self.logits = nn.Linear(prev, action_dim)
    def forward(self, x):
        h = self.body(x)
        return self.logits(h)

class CentralizedCritic(nn.Module):
    def __init__(self, input_dim, hidden=[256,256]):
        super().__init__()
        layers=[]
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev,h)); layers.append(nn.ReLU()); prev = h
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x).squeeze(-1)

Rollout = namedtuple('Rollout', ['obs','actions','log_probs','rewards','values','dones'])


class RolloutStorage:
    def __init__(self):
        self.obs=[]; self.actions=[]; self.log_probs=[]; self.rewards=[]; self.values=[]; self.dones=[]
    def add(self,obs,actions,log_probs,rewards,values,dones):
        self.obs.append(obs); self.actions.append(actions)
        self.log_probs.append(log_probs); self.rewards.append(rewards)
        self.values.append(values); self.dones.append(dones)
    def clear(self):
        self.obs=[]; self.actions=[]; self.log_probs=[]; self.rewards=[]; self.values=[]; self.dones=[]
    def sample(self):
        return Rollout(
            obs=torch.stack(self.obs),
            actions=torch.stack(self.actions),
            log_probs=torch.stack(self.log_probs),
            rewards=torch.stack(self.rewards),
            values=torch.stack(self.values),
            dones=torch.stack(self.dones)
        )


class MAPPOAgent:
    def __init__(self, num_agents, obs_dim, action_dim, state_dim):
        self.num_agents = num_agents
        self.action_dim = action_dim

        self.actors = nn.ModuleList([ActorNet(obs_dim, action_dim).to(device) for _ in range(num_agents)])
        self.critic = CentralizedCritic(state_dim).to(device)

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=POLICY_LR) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=VALUE_LR)

    def save(self, path):
        checkpoint = {
            'actor_states': [actor.state_dict() for actor in self.actors],
            'critic_state': self.critic.state_dict(),
            'actor_optim_states': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optim_state': self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, path)
        print(f" Model checkpoint saved at {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actor_states'][i])
        self.critic.load_state_dict(checkpoint['critic_state'])
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optim_states'][i])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state'])
        print(f" Model checkpoint loaded from {path}")


    def select_actions(self, obs_dict):
        actions = {}
        log_probs = {}
        values = {}
        obs_tensors = []
        for i, (tls, obs) in enumerate(sorted(obs_dict.items())):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            obs_tensors.append(obs_tensor)
            logits = self.actors[i](obs_tensor.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            actions[tls] = a.item()
            log_probs[tls] = dist.log_prob(a)
        state_input = torch.cat(obs_tensors).unsqueeze(0)
        value_t = self.critic(state_input).squeeze()
        for tls in sorted(obs_dict.keys()):
            values[tls] = value_t
        return actions, log_probs, values

    def evaluate_actions(self, obs_batch, action_batch):
        batch_size = obs_batch.size(0)
        num_agents = obs_batch.size(1)
        log_probs = []
        entropies = []
        values = []
        for i in range(num_agents):
            actor = self.actors[i]
            obs_i = obs_batch[:, i, :]
            logits = actor(obs_i)
            dist = torch.distributions.Categorical(logits=logits)
            a = action_batch[:, i]
            log_prob = dist.log_prob(a)
            entropy = dist.entropy()
            value = self.critic(obs_batch.view(batch_size, -1))  
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        values = values[0]  
        return log_probs, entropies, values

    def update(self, rollout):
        obs = rollout.obs.detach().to(device)
        actions = rollout.actions.detach().to(device)
        old_log_probs = rollout.log_probs.detach().to(device)
        rewards = rollout.rewards.detach().to(device)
        values = rollout.values.detach().to(device)
        dones = rollout.dones.detach().to(device)


        batch_size, num_agents, obs_dim = obs.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        for i in range(num_agents):
            gae = 0
            last_value = values[-1, i].item()
            for t in reversed(range(batch_size)):
                mask = 1.0 - dones[t, i].float()
                delta = rewards[t, i] + GAMMA * last_value * mask - values[t, i]
                gae = delta + GAMMA * GAE_LAMBDA * mask * gae
                advantages[t, i] = gae
                returns[t, i] = advantages[t, i] + values[t, i]
                last_value = values[t, i]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            for start in range(0, batch_size, batch_size//MINI_BATCHES):
                end = start + batch_size//MINI_BATCHES
                mb_obs = obs[start:end]
                mb_actions = actions[start:end]
                mb_old_log_probs = old_log_probs[start:end]
                mb_returns = returns[start:end]
                mb_advantages = advantages[start:end]

                log_probs = []
                entropies = []
                for i in range(num_agents):
                    logits = self.actors[i](mb_obs[:, i, :])
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(mb_actions[:, i])
                    entropy = dist.entropy()
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                log_probs = torch.stack(log_probs, dim=1)
                entropies = torch.stack(entropies, dim=1)

                flattened_obs = mb_obs.view(mb_obs.size(0), -1)
                values_pred = self.critic(flattened_obs)

                ratio = (log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, mb_returns.mean(dim=1))  

                entropy_loss = -entropies.mean()

                for i in range(num_agents):
                    self.actor_optimizers[i].zero_grad()
                self.critic_optimizer.zero_grad()

                total_loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(sum([list(actor.parameters()) for actor in self.actors], [])) + list(self.critic.parameters()),
                    MAX_GRAD_NORM)

                for i in range(num_agents):
                    self.actor_optimizers[i].step()
                self.critic_optimizer.step()

#training:
def train():
    os.makedirs("checkpoints", exist_ok=True)
    env = MultiAgentSumoEnv()
    label = "sim_main"
    tls_info = None

    episode_rewards = []
    episode_queue_lengths = []
    episode_waiting_times = []

    # Start SUMO
    env.start_sumo(sumo_cmd_template, label=label)
    tls_info = env.tls_info
    num_agents = len(tls_info)
    obs_dim = len(env.get_local_obs(next(iter(tls_info.keys()))))
    state_dim = num_agents * obs_dim
    action_dim = len(env.green_options)
    ppo_agent = MAPPOAgent(num_agents, obs_dim, action_dim, state_dim)

    resume_path = "checkpoints/ppo_latest.pth"
    if os.path.exists(resume_path):
        ppo_agent.load(resume_path)

    for ep in range(NUM_EPISODES):
        print(f"Starting Episode {ep+1}/{NUM_EPISODES}")
        env.close_sumo()
        env.start_sumo(sumo_cmd_template, label=label)
        env.reset_env_state()

        prev_metrics = {tls: None for tls in tls_info}
        ep_rewards = defaultdict(float)
        ep_queues = defaultdict(float)
        ep_waits = defaultdict(float)
        rollout_storage = RolloutStorage()
        done = False
        step = 0

        while not done and step < MAX_EPISODE_STEPS:
            obs = {tls: env.get_local_obs(tls) for tls in tls_info}
            actions, log_probs, values = ppo_agent.select_actions(obs)
            next_obs, rewards, done, new_metrics, _ = env.step(actions, prev_metrics)
            prev_metrics = new_metrics

            obs_tensor = torch.tensor(
                np.array([obs[tls] for tls in sorted(tls_info.keys())]),
                dtype=torch.float32, device=device
            )
            action_tensor = torch.tensor(
                [actions[tls] for tls in sorted(tls_info.keys())],
                dtype=torch.int64, device=device
            )
            log_prob_tensor = torch.stack([log_probs[tls] for tls in sorted(tls_info.keys())]).to(device)
            reward_tensor = torch.tensor(
                [rewards[tls] for tls in sorted(tls_info.keys())],
                dtype=torch.float32, device=device
            )
            value_tensor = torch.stack([values[tls].detach() for tls in sorted(tls_info.keys())]).to(device)
            done_tensor = torch.tensor([done]*num_agents, dtype=torch.float32, device=device)

            rollout_storage.add(obs_tensor, action_tensor, log_prob_tensor, reward_tensor, value_tensor, done_tensor)

            for tls in tls_info:
                ep_rewards[tls] += rewards[tls]
                ep_queues[tls] += new_metrics[tls]['queues']
                ep_waits[tls] += new_metrics[tls]['waiting']

            step += 1
            if step % ROLLOUT_STEPS == 0 or done:
                rollout = rollout_storage.sample()
                rollout = Rollout(
                    obs=rollout.obs.view(-1, num_agents, obs_dim),
                    actions=rollout.actions.view(-1, num_agents),
                    log_probs=rollout.log_probs.view(-1, num_agents),
                    rewards=rollout.rewards.view(-1, num_agents),
                    values=rollout.values.view(-1, num_agents),
                    dones=rollout.dones.view(-1, num_agents)
                )
                ppo_agent.update(rollout)
                rollout_storage.clear()

        avg_reward = np.mean(list(ep_rewards.values()))
        avg_queue = np.mean(list(ep_queues.values())) / max(1, step)
        avg_wait = np.mean(list(ep_waits.values())) / max(1, step)
        episode_rewards.append(avg_reward)
        episode_queue_lengths.append(avg_queue)
        episode_waiting_times.append(avg_wait)
        print(f"Episode {ep+1} Reward: {avg_reward:.3f} Queue: {avg_queue:.3f} Wait: {avg_wait:.3f}")


        if (ep + 1) % 1000 == 0:
            save_path = f"checkpoints/ppo_ep{ep+1}.pth"
            ppo_agent.save(save_path)
            ppo_agent.save("checkpoints/ppo_latest.pth")  
            metrics = {
                "rewards": episode_rewards,
                "queues": episode_queue_lengths,
                "waiting_times": episode_waiting_times
            }
            with open("checkpoints/training_metrics.json", "w") as f:
                json.dump(metrics, f)
            print(" Saved metrics & checkpoint.")

    
    ppo_agent.save("checkpoints/ppo_final_model.pth")
    env.close_sumo()
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, color='red')
    plt.title("Reward per Episode (Raw)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_queue_lengths, color='green')
    plt.title("Queue Length per Episode (Raw)")
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_waiting_times, color='blue')
    plt.title("Waiting Time per Episode (Raw)")
    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

    def smooth_curve(values, window_size=10):
        if len(values) < window_size:
            return values
        return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

    window_size = 20

    reward_window = smooth_curve(episode_rewards, window_size)
    queue_window = smooth_curve(episode_queue_lengths, window_size)
    wait_window  = smooth_curve(episode_waiting_times, window_size)
    
    plt.figure(figsize=(10,5))
    plt.plot(reward_window, color='red')
    plt.title(f"Reward per Episode (Moving Average, window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(queue_window, color='green')
    plt.title(f"Queue Length per Episode (Moving Average, window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(wait_window, color='blue')
    plt.title(f"Waiting Time per Episode (Moving Average, window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()


    from scipy.ndimage import gaussian_filter1d
    sigma = 2
    reward_gauss = gaussian_filter1d(episode_rewards, sigma=sigma)
    queue_gauss = gaussian_filter1d(episode_queue_lengths, sigma=sigma)
    wait_gauss  = gaussian_filter1d(episode_waiting_times, sigma=sigma)
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, color='lightcoral', alpha=0.3, label='Raw')
    plt.plot(reward_gauss, color='red', linewidth=2, label=f'Gaussian σ={sigma}')
    plt.title("Reward per Episode (Gaussian Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_queue_lengths, color='lightgreen', alpha=0.3, label='Raw')
    plt.plot(queue_gauss, color='green', linewidth=2, label=f'Gaussian σ={sigma}')
    plt.title("Queue Length per Episode (Gaussian Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_waiting_times, color='lightblue', alpha=0.3, label='Raw')
    plt.plot(wait_gauss, color='blue', linewidth=2, label=f'Gaussian σ={sigma}')
    plt.title("Waiting Time per Episode (Gaussian Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
train()