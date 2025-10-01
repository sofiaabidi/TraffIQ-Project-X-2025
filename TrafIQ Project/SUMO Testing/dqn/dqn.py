#standard libs
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import traci

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if 'SUMO_HOME' in os.environ:
    tools =  os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_CFG = "dqn.sumocfg"
STEP_LENGTH = 1.0
traci.start(["sumo", "-c", SUMO_CFG, "--step-length", str(STEP_LENGTH)])

#hyperparameters:

#dqn network:
class dqn(nn.module):
    def __init__(self, state_size, action_size):
        super(dqn, self).__init__()

        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
#experience replay:
class experience_replay:
    def __init__(self, capacity):
        self.memory =  deque(maxlen = capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class traffic_agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.policy_net = dqn(state_size, action_size)
        self.target_net = dqn(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state.dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = LEARNING_RATE)
        self.memory = experience_replay(REPLAY_MEMORY_SIZE)

        self.epsilon = EPSILON_START
    
    def choose_action(self, state):

        if random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
            
    def replay(self):
        