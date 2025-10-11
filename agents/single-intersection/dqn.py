import os
import sys
import platform
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from IPython.display import clear_output

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/home/jupyter-yashvi_mehta/sumo"  #sumo_home path here
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import traci  
print(f"SUMO_HOME set to: {os.environ['SUMO_HOME']}")

#change sumo_cfg paths if needed
sumo_binary = "/home/jupyter-yashvi_mehta/.local/bin/sumo" 
sumo_config_file = "environments\single-intersection\4x4-single-grid\single_grid.sumocfg"
sumo_cmd = [
    sumo_binary,
    "-c", sumo_config_file,
    "--step-length", "1.0",
    "--no-step-log", "true"
]

#hypyerparameters
episodes = 1000        
t_min, t_max, step = 15, 90, 5
green_actions = list(range(t_min, t_max+1, step))
yellow_time = 3
alpha = 1e-3               
gamma = 0.99                  
epsilon = 1.0
epsilon_decay = 0.9995      
epsilon_min = 0.05
MAX_STEPS = 3600

batch_size = 64             
target_update = 10         

directions = ['NS','EW']
actions = [(d,i) for d in range(len(directions)) for i in range(len(green_actions))]

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
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices])**(-beta)
        weights /= weights.max()
        return samples, weights, indices
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128,128,64]):
        super().__init__()
        layers=[]
        prev_dim=input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim,h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim=h
        layers.append(nn.Linear(prev_dim,output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self,x):
        return self.network(x)

prev_metrics = {}
current_phase_index = 0
current_simulation_step = 0
last_switch_step = 0

def get_queue_length(lane_ids):
    return sum(traci.lane.getLastStepHaltingNumber(l) for l in lane_ids)

def get_comprehensive_metrics():
    vehicle_ids = traci.vehicle.getIDList()
    if not vehicle_ids:
        return {'total_vehicles':0,'avg_waiting_time':0,'avg_speed':0,'total_co2':0,'total_fuel':0,'throughput':0,'queue_lengths':[0,0,0,0]}
    total_waiting=sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
    total_speed=sum(traci.vehicle.getSpeed(vid) for vid in vehicle_ids)
    total_co2=sum(traci.vehicle.getCO2Emission(vid) for vid in vehicle_ids)
    total_fuel=sum(traci.vehicle.getFuelConsumption(vid) for vid in vehicle_ids)
    q_EB=get_queue_length(["Node1_2_EB_0","Node1_2_EB_1","Node1_2_EB_2"])
    q_SB=get_queue_length(["Node2_7_SB_0","Node2_7_SB_1","Node2_7_SB_2"])
    q_WB=get_queue_length(["Node2_3_WB_0","Node2_3_WB_1","Node2_3_WB_2"])
    q_NB=get_queue_length(["Node2_5_NB_0","Node2_5_NB_1","Node2_5_NB_2"])
    return {'total_vehicles':len(vehicle_ids),
            'avg_waiting_time':total_waiting/len(vehicle_ids),
            'avg_speed':total_speed/len(vehicle_ids),
            'total_co2':total_co2,
            'total_fuel':total_fuel,
            'throughput':len(vehicle_ids),
            'queue_lengths':[q_EB,q_SB,q_WB,q_NB]}

def get_enhanced_state():
    global current_simulation_step,last_switch_step,current_phase_index
    metrics=get_comprehensive_metrics()
    normalized_queues=[min(q/20.0,1.0) for q in metrics['queue_lengths']]
    normalized_waiting=min(metrics['avg_waiting_time']/300.0,1.0)
    normalized_speed=metrics['avg_speed']/50.0
    normalized_vehicles=min(metrics['total_vehicles']/100.0,1.0)
    time_since_switch=min((current_simulation_step-last_switch_step)/60.0,1.0)
    phase_encoding=[1.0,0.0] if current_phase_index==0 else [0.0,1.0]
    return np.array(normalized_queues+[normalized_waiting,normalized_speed,normalized_vehicles,time_since_switch]+phase_encoding,dtype=np.float32)

def get_multi_objective_reward():
    global prev_metrics
    current_metrics=get_comprehensive_metrics()
    if not prev_metrics:
        prev_metrics=current_metrics
        return 0.0
    weights={'waiting_time':-2.0,'queue_length':-1.0,'co2_emissions':-1.5,'fuel_consumption':-1.0,'throughput':2.0,'speed':1.0}
    waiting_improvement=prev_metrics['avg_waiting_time']-current_metrics['avg_waiting_time']
    queue_improvement=sum(prev_metrics['queue_lengths'])-sum(current_metrics['queue_lengths'])
    co2_improvement=prev_metrics['total_co2']-current_metrics['total_co2']
    fuel_improvement=prev_metrics['total_fuel']-current_metrics['total_fuel']
    throughput_improvement=current_metrics['throughput']-prev_metrics['throughput']
    speed_improvement=current_metrics['avg_speed']-prev_metrics['avg_speed']
    reward=(weights['waiting_time']*waiting_improvement+
            weights['queue_length']*queue_improvement+
            weights['co2_emissions']*co2_improvement/1000.0+
            weights['fuel_consumption']*fuel_improvement/1000.0+
            weights['throughput']*throughput_improvement+
            weights['speed']*speed_improvement)
    max_queue=max(current_metrics['queue_lengths'])
    if max_queue>15: reward-=(max_queue-15)*0.5
    prev_metrics=current_metrics
    return reward

def apply_enhanced_action(action_idx,tls_id="Node2"):
    global current_phase_index,current_simulation_step,last_switch_step
    direction,duration_idx=actions[action_idx]
    green_time=green_actions[duration_idx]
    target_phase=0 if direction==0 else 2
    if target_phase!=current_phase_index and (current_simulation_step-last_switch_step)>=15:
        current_yellow_phase=current_phase_index+1
        traci.trafficlight.setPhase(tls_id,current_yellow_phase)
        for _ in range(3):
            if traci.simulation.getMinExpectedNumber()>0:
                traci.simulationStep()
                current_simulation_step+=1
        current_phase_index=target_phase
        last_switch_step=current_simulation_step
    traci.trafficlight.setPhase(tls_id,current_phase_index)
    for _ in range(green_actions[duration_idx]):
        if traci.simulation.getMinExpectedNumber()>0:
            traci.simulationStep()
            current_simulation_step+=1

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim=10
action_dim=len(actions)
policy_net=DQN(state_dim,action_dim).to(device)
target_net=DQN(state_dim,action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=alpha, weight_decay=1e-5)
replay_buffer=ReplayBuffer(50000)
batch_size=128
target_update=50

ep_history,reward_history=[],[]
metrics_history={'queue_length':[],'waiting_time':[],'co2_emissions':[],'throughput':[]}

for ep in range(episodes):
    traci.start(sumo_cmd,label=f"sim{ep}")
    prev_metrics={}
    current_phase_index=0
    current_simulation_step=0
    last_switch_step=0
    for _ in range(10):
        traci.simulationStep()
        current_simulation_step+=1
    state=get_enhanced_state()
    ep_reward=0.0
    ep_metrics={'queue':[],'waiting':[],'co2':[],'throughput':[]}

    while traci.simulation.getMinExpectedNumber()>0 and current_simulation_step<MAX_STEPS:
        state_tensor=torch.FloatTensor(state).unsqueeze(0).to(device)
        if random.random()<epsilon:
            action=random.randint(0,action_dim-1)
        else:
            with torch.no_grad():
                q_vals=policy_net(state_tensor)
                action=int(torch.argmax(q_vals).item())
        apply_enhanced_action(action)
        next_state=get_enhanced_state()
        reward=get_multi_objective_reward()
        ep_reward+=reward
        replay_buffer.push((state,action,reward,next_state))

        if len(replay_buffer)>=batch_size:
            sample_result=replay_buffer.sample(batch_size)
            if sample_result is not None:
                experiences,weights,indices=sample_result
                s,a,r,s_next=zip(*experiences)
                s=torch.FloatTensor(np.array(s)).to(device)
                a=torch.LongTensor(a).unsqueeze(1).to(device)
                r=torch.FloatTensor(r).unsqueeze(1).to(device)
                s_next=torch.FloatTensor(np.array(s_next)).to(device)
                q_values=policy_net(s).gather(1,a)
                next_q_values=target_net(s_next).max(1)[0].unsqueeze(1).detach()
                expected_q=r+gamma*next_q_values
                td_errors=torch.abs(q_values-expected_q).detach().cpu().numpy().flatten()
                replay_buffer.update_priorities(indices,td_errors+1e-6)
                loss = F.smooth_l1_loss(q_values, expected_q)  # Huber Loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(),1.0)
                optimizer.step()

        current_metrics=get_comprehensive_metrics()
        ep_metrics['queue'].append(sum(current_metrics['queue_lengths']))
        ep_metrics['waiting'].append(current_metrics['avg_waiting_time'])
        ep_metrics['co2'].append(current_metrics['total_co2'])
        ep_metrics['throughput'].append(current_metrics['throughput'])
        state=next_state

    if (ep+1)%target_update==0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon=max(epsilon_min,epsilon*epsilon_decay)
    ep_history.append(ep+1)
    reward_history.append(ep_reward)
    metrics_history['queue_length'].append(np.mean(ep_metrics['queue']))
    metrics_history['waiting_time'].append(np.mean(ep_metrics['waiting']))
    metrics_history['co2_emissions'].append(np.mean(ep_metrics['co2']))
    metrics_history['throughput'].append(np.mean(ep_metrics['throughput']))

    clear_output(wait=True)
    from scipy.ndimage import gaussian_filter1d

    sigma = 2      
    smoothed_rewards = gaussian_filter1d(reward_history, sigma=sigma)
    smoothed_queue = gaussian_filter1d(metrics_history['queue_length'], sigma=sigma)
    smoothed_wait = gaussian_filter1d(metrics_history['waiting_time'], sigma=sigma)
    
    print(f"Episode {ep+1}/{episodes} | Reward={ep_reward:.2f} | "
          f"Avg Queue={np.mean(ep_metrics['queue']):.1f} | "
          f"Avg Wait={np.mean(ep_metrics['waiting']):.1f} | "
          f"Epsilon={epsilon:.3f}")
    
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,2,1)
    plt.plot(ep_history, reward_history, alpha=0.3, label='Raw Reward')
    plt.plot(ep_history, smoothed_rewards, color='red', label='Smoothed Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(ep_history, metrics_history['queue_length'], alpha=0.3, label='Raw Queue Length')
    plt.plot(ep_history, smoothed_queue, color='blue', label='Smoothed Queue Length')
    plt.plot(ep_history, metrics_history['waiting_time'], alpha=0.3, label='Raw Waiting Time')
    plt.plot(ep_history, smoothed_wait, color='green', label='Smoothed Waiting Time')
    plt.xlabel('Episode')
    plt.grid(True)
    plt.legend()
    
    plt.show()