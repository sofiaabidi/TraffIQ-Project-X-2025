import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci  

sumo_config = [
    "sumo",                 
    "-c", "dqn.sumocfg",
    "--step-length", "1.0",
    "--no-step-log", "true",
    "--no-warnings", "true",
]

episodes = 1000   
t_min, t_max, step = 10, 60, 5
green_actions = list(range(t_min, t_max + 1, step))  
yellow_time = 3
alpha = 0.05
gamma = 0.999
epsilon = 1.0
epsilon_decay = 0.9999
epsilon_min = 0.01

actions = list(range(len(green_actions)))  

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Global vars
prev_wait = 0
prev_queue = 0
current_phase_index = 0
current_simulation_step = 0
last_switch_step = 0

def get_queue_length(lane_ids):
    return sum(traci.lane.getLastStepHaltingNumber(l) for l in lane_ids)

def get_state():
    q_EB = get_queue_length(["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2"])
    q_SB = get_queue_length(["Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"])
    q_WB = get_queue_length(["Node2_3_WB_0", "Node2_3_WB_1", "Node2_3_WB_2"])
    q_NB = get_queue_length(["Node2_5_NB_0", "Node2_5_NB_1", "Node2_5_NB_2"])
    return np.array([q_EB, q_SB, q_WB, q_NB, current_phase_index], dtype=np.float32)

def get_reward():
    global prev_wait, prev_queue
    vehicle_ids = traci.vehicle.getIDList()
    current_wait = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
    current_queue = (
        get_queue_length(["Node1_2_EB_0","Node1_2_EB_1","Node1_2_EB_2"]) +
        get_queue_length(["Node2_7_SB_0","Node2_7_SB_1","Node2_7_SB_2"]) +
        get_queue_length(["Node2_3_WB_0","Node2_3_WB_1","Node2_3_WB_2"]) +
        get_queue_length(["Node2_5_NB_0","Node2_5_NB_1","Node2_5_NB_2"])
    )
    reward = (prev_wait - current_wait) + 0.5 * (prev_queue - current_queue)
    prev_wait, prev_queue = current_wait, current_queue
    return reward

def apply_action(action, tls_id="Node2"):
    global current_phase_index, current_simulation_step, last_switch_step
    green_time = green_actions[action]
    phase = [0, 2] 
    chosen_phase = phase[current_phase_index]

    traci.trafficlight.setPhase(tls_id, chosen_phase)
    for _ in range(green_time):
        traci.simulationStep()
        current_simulation_step += 1

    traci.trafficlight.setPhase(tls_id, chosen_phase + 1)
    for _ in range(yellow_time):
        traci.simulationStep()
        current_simulation_step += 1

    current_phase_index = 1 - current_phase_index
    last_switch_step = current_simulation_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = 5 
action_dim = len(actions)

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
replay_buffer = ReplayBuffer(100000)
batch_size = 64
target_update = 100  

ep_history = []
reward_history = []
queue_history = []

print("\nTraining the Model:")

for ep in range(episodes):
    traci.start(sumo_config)

    prev_wait, prev_queue = 0, 0
    current_phase_index = 0

    for _ in range(5):
        traci.simulationStep()

    state = get_state()
    ep_reward = 0.0

    while traci.simulation.getMinExpectedNumber() > 0:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            with torch.no_grad():
                q_vals = policy_net(state_tensor)
                action = int(torch.argmax(q_vals).item())

        apply_action(action)
        next_state = get_state()
        reward = get_reward()
        ep_reward += reward

        replay_buffer.push((state, action, reward, next_state))
        state = next_state

        if len(replay_buffer) >= batch_size:
            s, a, r, s_next = zip(*replay_buffer.sample(batch_size))
            s = torch.FloatTensor(np.array(s)).to(device)
            a = torch.LongTensor(a).unsqueeze(1).to(device)
            r = torch.FloatTensor(r).unsqueeze(1).to(device)
            s_next = torch.FloatTensor(np.array(s_next)).to(device)
            q_values      = policy_net(s).gather(1, a)
            next_q_values = target_net(s_next).max(1)[0].unsqueeze(1).detach()
            expected_q    = r + gamma * next_q_values

            loss = F.mse_loss(q_values, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if (ep + 1) % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    ep_history.append(ep+1)
    reward_history.append(ep_reward)
    queue_history.append(sum(state[:-1]))
    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1}/{episodes} | Reward={ep_reward:.2f} | eps={epsilon:.3f}")
    traci.close()

print("\nTraining completed:")

plt.figure(figsize=(10,6))
plt.plot(ep_history, reward_history, label="Episode Reward", alpha=0.6)

window = 50
if len(reward_history) >= window:
    moving_avg = np.convolve(reward_history, np.ones(window)/window, mode="valid")
    plt.plot(ep_history[window-1:], moving_avg, label=f"Moving Avg ({window} eps)", color="red")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rewards.png")

plt.figure(figsize=(10,6))
plt.plot(ep_history, queue_history, marker='.', alpha=0.7, label="Total Queue Length")

if len(queue_history) >= window:
    moving_avg_q = np.convolve(queue_history, np.ones(window)/window, mode="valid")
    plt.plot(ep_history[window-1:], moving_avg_q, label=f"Moving Avg ({window} eps)", color="orange")

plt.xlabel("Episode")
plt.ylabel("Queue Length")
plt.title("Queue Length per Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("queues.png")