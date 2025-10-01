import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

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
    "--waiting-time-memory", "10000",
    "--time-to-teleport", "-1"
]

#hyperparameters
episodes = 2000
t_min, t_max, step = 15, 90, 5
green_actions = list(range(t_min, t_max + 1, step))
yellow_time = 3
alpha = 0.0001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995 
epsilon_min = 0.05

#action space
directions = ['NS', 'EW']  
actions = []
for direction in range(len(directions)):
    for duration_idx in range(len(green_actions)):
        actions.append((direction, duration_idx))

class ReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, transition):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
            
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128, 64]):
        super(ImprovedDQN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Global variables
prev_metrics = {}
current_phase_index = 0
current_simulation_step = 0
last_switch_step = 0

def get_comprehensive_metrics():
    """Get comprehensive traffic metrics for better state representation and rewards"""
    vehicle_ids = traci.vehicle.getIDList()
    
    if not vehicle_ids:
        return {
            'total_vehicles': 0,
            'avg_waiting_time': 0,
            'avg_speed': 0,
            'total_co2': 0,
            'total_fuel': 0,
            'throughput': 0,
            'queue_lengths': [0, 0, 0, 0]
        }
    
    # Traffic metrics
    total_waiting = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
    total_speed = sum(traci.vehicle.getSpeed(vid) for vid in vehicle_ids)
    total_co2 = sum(traci.vehicle.getCO2Emission(vid) for vid in vehicle_ids)
    total_fuel = sum(traci.vehicle.getFuelConsumption(vid) for vid in vehicle_ids)
    
    # Queue lengths for each direction
    q_EB = get_queue_length(["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2"])
    q_SB = get_queue_length(["Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"])
    q_WB = get_queue_length(["Node2_3_WB_0", "Node2_3_WB_1", "Node2_3_WB_2"])
    q_NB = get_queue_length(["Node2_5_NB_0", "Node2_5_NB_1", "Node2_5_NB_2"])
    
    return {
        'total_vehicles': len(vehicle_ids),
        'avg_waiting_time': total_waiting / len(vehicle_ids),
        'avg_speed': total_speed / len(vehicle_ids),
        'total_co2': total_co2,
        'total_fuel': total_fuel,
        'throughput': len(vehicle_ids),
        'queue_lengths': [q_EB, q_SB, q_WB, q_NB]
    }

def get_queue_length(lane_ids):
    return sum(traci.lane.getLastStepHaltingNumber(l) for l in lane_ids)

def get_enhanced_state():
    """Enhanced state representation with normalized features"""
    metrics = get_comprehensive_metrics()
    
    # Normalize queue lengths (assuming max 20 vehicles per direction)
    normalized_queues = [min(q/20.0, 1.0) for q in metrics['queue_lengths']]
    
    # Normalize other metrics
    normalized_waiting = min(metrics['avg_waiting_time'] / 300.0, 1.0)  # max 5 minutes
    normalized_speed = metrics['avg_speed'] / 50.0  # max speed 50 km/h
    normalized_vehicles = min(metrics['total_vehicles'] / 100.0, 1.0)  # max 100 vehicles
    
    # Time since last phase change (normalized)
    time_since_switch = min((current_simulation_step - last_switch_step) / 60.0, 1.0)
    
    # Current phase one-hot encoding
    phase_encoding = [1.0, 0.0] if current_phase_index == 0 else [0.0, 1.0]
    
    state = (normalized_queues + 
             [normalized_waiting, normalized_speed, normalized_vehicles, time_since_switch] +
             phase_encoding)
    
    return np.array(state, dtype=np.float32)

def get_multi_objective_reward():
    """Multi-objective reward function considering multiple traffic optimization goals"""
    global prev_metrics
    
    current_metrics = get_comprehensive_metrics()
    
    if not prev_metrics:
        prev_metrics = current_metrics
        return 0.0
    
    # Reward components with weights
    weights = {
        'waiting_time': -2.0,    # Minimize waiting time
        'queue_length': -1.0,    # Minimize queue length
        'co2_emissions': -1.5,   # Minimize CO2 emissions
        'fuel_consumption': -1.0, # Minimize fuel consumption
        'throughput': 2.0,       # Maximize throughput
        'speed': 1.0             # Maximize average speed
    }
    
    # Calculate improvements (negative = improvement for costs)
    waiting_improvement = prev_metrics['avg_waiting_time'] - current_metrics['avg_waiting_time']
    queue_improvement = sum(prev_metrics['queue_lengths']) - sum(current_metrics['queue_lengths'])
    co2_improvement = prev_metrics['total_co2'] - current_metrics['total_co2']
    fuel_improvement = prev_metrics['total_fuel'] - current_metrics['total_fuel']
    throughput_improvement = current_metrics['throughput'] - prev_metrics['throughput']
    speed_improvement = current_metrics['avg_speed'] - prev_metrics['avg_speed']
    
    # Calculate weighted reward
    reward = (weights['waiting_time'] * waiting_improvement +
              weights['queue_length'] * queue_improvement +
              weights['co2_emissions'] * co2_improvement / 1000.0 +  # Scale CO2
              weights['fuel_consumption'] * fuel_improvement / 1000.0 +  # Scale fuel
              weights['throughput'] * throughput_improvement +
              weights['speed'] * speed_improvement)
    
    # Penalty for very long queues
    max_queue = max(current_metrics['queue_lengths'])
    if max_queue > 15:
        reward -= (max_queue - 15) * 0.5
    
    # Update previous metrics
    prev_metrics = current_metrics
    
    return reward

def apply_enhanced_action(action_idx, tls_id="Node2"):
    """Enhanced action application with proper phase management"""
    global current_phase_index, current_simulation_step, last_switch_step
    
    direction, duration_idx = actions[action_idx]
    green_time = green_actions[duration_idx]
    
    # Phase mapping: 0=NS green, 2=EW green
    target_phase = 0 if direction == 0 else 2
    
    # Only switch if different from current phase and minimum time elapsed
    if target_phase != current_phase_index and (current_simulation_step - last_switch_step) >= 15:
        # Yellow phase transition
        current_yellow_phase = current_phase_index + 1
        traci.trafficlight.setPhase(tls_id, current_yellow_phase)
        
        for _ in range(yellow_time):
            if traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                current_simulation_step += 1
        
        current_phase_index = target_phase
        last_switch_step = current_simulation_step
    
    # Apply green phase
    traci.trafficlight.setPhase(tls_id, current_phase_index)
    
    for _ in range(green_time):
        if traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            current_simulation_step += 1

# Enhanced DQN setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = 10  # Enhanced state dimension
action_dim = len(actions)

policy_net = ImprovedDQN(state_dim, action_dim).to(device)
target_net = ImprovedDQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha, weight_decay=1e-5)
replay_buffer = ReplayBuffer(50000)
batch_size = 64
target_update = 50
beta_start = 0.4
beta_frames = 100000

# Training metrics
ep_history = []
reward_history = []
metrics_history = {
    'queue_length': [],
    'waiting_time': [],
    'co2_emissions': [],
    'throughput': []
}

print("\nTraining Enhanced DQN Model:")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Device: {device}")

for ep in range(episodes):
    traci.start(sumo_config)
    
    # Reset environment
    prev_metrics = {}
    current_phase_index = 0
    current_simulation_step = 0
    last_switch_step = 0
    
    # Warm up simulation
    for _ in range(10):
        traci.simulationStep()
        current_simulation_step += 1
    
    state = get_enhanced_state()
    ep_reward = 0.0
    ep_metrics = {'queue': [], 'waiting': [], 'co2': [], 'throughput': []}
    
    while traci.simulation.getMinExpectedNumber() > 0:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q_vals = policy_net(state_tensor)
                action = int(torch.argmax(q_vals).item())
        
        # Apply action
        apply_enhanced_action(action)
        next_state = get_enhanced_state()
        reward = get_multi_objective_reward()
        ep_reward += reward
        
        # Store experience
        replay_buffer.push((state, action, reward, next_state))
        
        # Training step
        if len(replay_buffer) >= batch_size:
            beta = min(1.0, beta_start + ep * (1.0 - beta_start) / beta_frames)
            sample_result = replay_buffer.sample(batch_size, beta)
            
            if sample_result is not None:
                experiences, weights, indices = sample_result
                
                s, a, r, s_next = zip(*experiences)
                s = torch.FloatTensor(np.array(s)).to(device)
                a = torch.LongTensor(a).unsqueeze(1).to(device)
                r = torch.FloatTensor(r).unsqueeze(1).to(device)
                s_next = torch.FloatTensor(np.array(s_next)).to(device)
                w = torch.FloatTensor(weights).unsqueeze(1).to(device)
                
                # Compute loss
                q_values = policy_net(s).gather(1, a)
                next_q_values = target_net(s_next).max(1)[0].unsqueeze(1).detach()
                expected_q = r + gamma * next_q_values
                
                td_errors = torch.abs(q_values - expected_q).detach().cpu().numpy().flatten()
                replay_buffer.update_priorities(indices, td_errors + 1e-6)
                
                loss = (w * F.mse_loss(q_values, expected_q, reduction='none')).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        # Collect metrics
        current_metrics = get_comprehensive_metrics()
        ep_metrics['queue'].append(sum(current_metrics['queue_lengths']))
        ep_metrics['waiting'].append(current_metrics['avg_waiting_time'])
        ep_metrics['co2'].append(current_metrics['total_co2'])
        ep_metrics['throughput'].append(current_metrics['throughput'])
        
        state = next_state
    
    if (ep + 1) % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    #epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    ep_history.append(ep + 1)
    reward_history.append(ep_reward)
    metrics_history['queue_length'].append(np.mean(ep_metrics['queue']))
    metrics_history['waiting_time'].append(np.mean(ep_metrics['waiting']))
    metrics_history['co2_emissions'].append(np.mean(ep_metrics['co2']))
    metrics_history['throughput'].append(np.mean(ep_metrics['throughput']))
    
    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{episodes} | "
              f"Reward={ep_reward:.2f} | "
              f"Avg Queue={np.mean(ep_metrics['queue']):.1f} | "
              f"Avg Wait={np.mean(ep_metrics['waiting']):.1f} | "
              f"Epsilon={epsilon:.3f}")
    traci.close()

print("\nDQN Training completed!")

torch.save(policy_net.state_dict(), 'dqn_policy.pth')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

#reward plotting
axes[0,0].plot(ep_history, reward_history, alpha=0.6, label="Episode Reward")
window = 100
if len(reward_history) >= window:
    moving_avg = np.convolve(reward_history, np.ones(window)/window, mode="valid")
    axes[0,0].plot(ep_history[window-1:], moving_avg, label=f"Moving Avg ({window})", color="red")
axes[0,0].set_xlabel("Episode")
axes[0,0].set_ylabel("Reward")
axes[0,0].set_title("Training Rewards")
axes[0,0].legend()
axes[0,0].grid(True)

#queue length
axes[0,1].plot(ep_history, metrics_history['queue_length'], alpha=0.6, label="Queue Length")
if len(metrics_history['queue_length']) >= window:
    moving_avg = np.convolve(metrics_history['queue_length'], np.ones(window)/window, mode="valid")
    axes[0,1].plot(ep_history[window-1:], moving_avg, label=f"Moving Avg ({window})", color="orange")
axes[0,1].set_xlabel("Episode")
axes[0,1].set_ylabel("Avg Queue Length")
axes[0,1].set_title("Queue Length Optimization")
axes[0,1].legend()
axes[0,1].grid(True)

#waiting time
axes[0,2].plot(ep_history, metrics_history['waiting_time'], alpha=0.6, label="Waiting Time")
if len(metrics_history['waiting_time']) >= window:
    moving_avg = np.convolve(metrics_history['waiting_time'], np.ones(window)/window, mode="valid")
    axes[0,2].plot(ep_history[window-1:], moving_avg, label=f"Moving Avg ({window})", color="green")
axes[0,2].set_xlabel("Episode")
axes[0,2].set_ylabel("Avg Waiting Time (s)")
axes[0,2].set_title("Waiting Time Optimization")
axes[0,2].legend()
axes[0,2].grid(True)

#co2 emissions
axes[1,0].plot(ep_history, metrics_history['co2_emissions'], alpha=0.6, label="CO2 Emissions")
if len(metrics_history['co2_emissions']) >= window:
    moving_avg = np.convolve(metrics_history['co2_emissions'], np.ones(window)/window, mode="valid")
    axes[1,0].plot(ep_history[window-1:], moving_avg, label=f"Moving Avg ({window})", color="brown")
axes[1,0].set_xlabel("Episode")
axes[1,0].set_ylabel("Avg CO2 Emissions")
axes[1,0].set_title("CO2 Emissions Optimization")
axes[1,0].legend()
axes[1,0].grid(True)

#throughput
axes[1,1].plot(ep_history, metrics_history['throughput'], alpha=0.6, label="Throughput")
if len(metrics_history['throughput']) >= window:
    moving_avg = np.convolve(metrics_history['throughput'], np.ones(window)/window, mode="valid")
    axes[1,1].plot(ep_history[window-1:], moving_avg, label=f"Moving Avg ({window})", color="purple")
axes[1,1].set_xlabel("Episode")
axes[1,1].set_ylabel("Avg Throughput")
axes[1,1].set_title("Throughput Optimization")
axes[1,1].legend()
axes[1,1].grid(True)

#final metrics
final_metrics = [
    np.mean(metrics_history['queue_length'][-50:]),
    np.mean(metrics_history['waiting_time'][-50:]),
    np.mean(metrics_history['co2_emissions'][-50:]),
    np.mean(metrics_history['throughput'][-50:])
]
metric_names = ['Queue Length', 'Waiting Time', 'CO2 Emissions', 'Throughput']
axes[1,2].bar(metric_names, final_metrics)
axes[1,2].set_title("Final Performance Metrics (Last 50 Episodes)")
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("dqn_results.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFinal Performance Summary:")
print(f"Average Queue Length (last 50 eps): {final_metrics[0]:.2f}")
print(f"Average Waiting Time (last 50 eps): {final_metrics[1]:.2f}s")
print(f"Average CO2 Emissions (last 50 eps): {final_metrics[2]:.2f}")
print(f"Average Throughput (last 50 eps): {final_metrics[3]:.2f}")