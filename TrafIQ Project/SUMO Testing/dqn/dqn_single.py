import os, sys, time, random, math, collections
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import traci

if not os.path.exists("results"):
    os.makedirs("results")
    
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please add SUMO_HOME to environment variables")

#sumo config:
sumoBinary = "sumo"
sumoCmd = [sumoBinary, "-c", "dqn.sumocfg", "--summary-output", "results/summary.xml", "--ignore-route-errors"]

traffic_light_id = "Node2"
detectors = ["EB", "WB", "NB", "SB"]

#rl hyperparameters:
episodes = 30
end_step = 2000
min_batch = 32
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

#state space:
state_size = 1 + len(detectors)

#action space:
min_green = 5
max_green = 45
step = 5
action_durations = list(range(min_green, max_green, step))
action_num = len(action_durations)

#dqn model:
class dqn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(dqn, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
policy_net = dqn(state_size, 64, action_num)
target_net = dqn(state_size, 64, action_num)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()

#replay memory
memory = []

def remember(state, action, reward, next_action, done):
    memory.append((state, action, reward, next_action, done))
    if len(memory) > 10000:
        memory.pop(0)

#defining state and reward:
def get_state(phase):
    state = [phase % 2]
    for det in detectors:
        q = traci.lanearea.getLastStepVehicleNumber(det)
        state.append(q)
    return np.array(state, dtype = np.float32)

#reward = -ve(total halting vehicles)
def get_reward():
    total_wait = 0
    for det in detectors:
        total_wait += traci.lanearea.getLastStepHaltingNumber(det)
    return -total_wait

#action selection:
def choose_action(state):
    global epsilon
    if random.random() < epsilon:
        return random.randint(0, action_num - 1)
    state_tensor = autograd.Variable(torch.FloatTensor(state)).unsqueze(0)
    q_values = policy_net(state_tensor)
    return torch.argmax(q_values).item()

#replay:
def replay():
    if len(memory) < min_batch:
        return
    batch = random.sample(memory, min_batch)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    targets = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_function(q_values, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============== TRAINING LOOP =================
for episode in range(episodes):
    traci.start(sumoCmd)
    step = 0
    total_reward = 0

    # initial state
    phase = traci.trafficlight.getPhase(traffic_light_id)
    state = get_state(phase)

    while step < end_step:
        # select action
        action = choose_action(state)
        green_time = action_durations[action]

        # apply action: set phase duration
        traci.trafficlight.setPhase(traffic_light_id, phase)
        traci.trafficlight.setPhaseDuration(traffic_light_id, green_time)

        # advance simulation
        for _ in range(green_time):
            traci.simulationStep()
            step += 1
            if step >= end_step:
                break

        # next state
        phase = traci.trafficlight.getPhase(traffic_light_id)
        next_state = get_state(phase)
        reward = get_reward()
        done = (step >= end_step)

        # store in memory
        remember(state, action, reward, next_state, done)

        # learn
        replay()

        state = next_state
        total_reward += reward

        if step % 100 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    traci.close()

    # decay exploration
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode+1}/{episodes}, Reward: {total_reward}")