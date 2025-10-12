import os  
import sys 
import random
import numpy as np
import matplotlib.pyplot as plt 

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci  

Sumo_config = [
    'sumo-gui',
    '-c', 'dqn.sumocfg',
    '--step-length', '0.10',
    '--delay','200',
    '--lateral-resolution','0'
]

traci.start("environments\single-intersection\4x4-single-grid\single_grid.sumocfg")
traci.gui.setSchema("View #0","real world")

q_EB_0=0
q_EB_1=0
q_EB_2=0
q_SB_0=0
q_SB_1=0
q_SB_2=0
q_NB_0=0
q_NB_1=0
q_NB_2=0
q_WB_0=0
q_WB_1=0
q_WB_2=0

current_phase=0

TOTAL_STEPS=10000

ALPHA=0.3
GAMMA=0.9                                                                  
EPSILON=1         
ACTIONS=[0,1,2,3]

Q_table = {}

MIN_GREEN_STEPS=100
last_switch_step=-MIN_GREEN_STEPS

ACTION_TO_PHASE = {
    0:0,  #EB/WB green
    1:2,  #SB green
    2:4,  #WB green
    3:6   #NB green
}
def get_max_Q_value_of_state(s): 
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

prev_avg_wait = 0 

def get_average_waiting_time():
    veh_ids = traci.vehicle.getIDList()
    if len(veh_ids) == 0:
        return 0
    total_wait = sum(traci.vehicle.getWaitingTime(vid) for vid in veh_ids)
    return total_wait / len(veh_ids)

def get_reward():
    global prev_avg_wait
    curr_avg_wait = get_average_waiting_time()
    reward = prev_avg_wait - curr_avg_wait
    prev_avg_wait = curr_avg_wait
    return reward


def make_state_discrete(x):
    return min(x//3,5)  

def get_state():  
    global q_EB_0,q_EB_1,q_EB_2,q_SB_0,q_SB_1,q_SB_2,q_WB_0,q_WB_1,q_WB_2,q_NB_0,q_NB_1,q_NB_2,current_phase
    traffic_light_id="Node2"
    
    q_EB_0=get_queue_length("Node1_2_EB_0")
    q_EB_1=get_queue_length("Node1_2_EB_1")
    q_EB_2=get_queue_length("Node1_2_EB_2")
    
    q_SB_0=get_queue_length("Node2_7_SB_0")
    q_SB_1=get_queue_length("Node2_7_SB_1")
    q_SB_2=get_queue_length("Node2_7_SB_2")

    q_WB_0=get_queue_length("Node2_3_WB_0")
    q_WB_1=get_queue_length("Node2_3_WB_1")
    q_WB_2=get_queue_length("Node2_3_WB_2")
    
    q_NB_0 = get_queue_length("Node2_5_NB_0")
    q_NB_1 = get_queue_length("Node2_5_NB_1")
    q_NB_2 = get_queue_length("Node2_5_NB_2")
    
    
    current_phase = get_current_phase(traffic_light_id)
    
    return (make_state_discrete(q_EB_0+q_EB_1+q_EB_2),make_state_discrete(q_SB_0+q_SB_1+q_SB_2),make_state_discrete(q_WB_0+q_WB_1+q_WB_2), make_state_discrete(q_NB_0+q_NB_1+q_NB_2), current_phase)

def apply_action(action, tls_id="Node2"): 
    global last_switch_step

    target_phase=ACTION_TO_PHASE[action]
    if target_phase!= get_current_phase(tls_id):
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program=traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases=len(program.phases)
            yellow_phase=get_current_phase(tls_id)+1
            traci.trafficlight.setPhase(tls_id,yellow_phase)
            traci.simulationStep() 
            traci.simulationStep() 
            next_phase= (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step

def update_Q_table(old_state, action, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)



def get_action_from_policy(state): 
    if random.random() < EPSILON:
        return random.choice(range(len(ACTIONS)))
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_table[state]))


def get_queue_length(detector_id): 
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

step_history=[]
reward_history=[]
queue_history=[]
cumulative_reward=0.0

print("\nStarting Learning")

for step in range(TOTAL_STEPS):
    current_simulation_step = step
    state = get_state()
    if state[-1]%2!=0:

        traci.simulationStep()
    else:
        action=get_action_from_policy(state)
        apply_action(action)
        for i in range(5):
            traci.simulationStep()
            step+=5  
        
        new_state=get_state()
        reward=get_reward()
        cumulative_reward+=reward
        
        update_Q_table(state, action, reward, new_state)
        updated_q_vals = Q_table[state]
    EPSILON=max(0.01,EPSILON*0.9995)

    #store data every 100 steps
    if step % 100 == 0:
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values(current_state): {updated_q_vals}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        cumulative_reward=0
        queue_history.append(sum(new_state[:-1])) 
        print("Current Q-table:")
        for st, qvals in Q_table.items():
            print(f"  {st} -> {qvals}")
traci.close()

#Q-Table
print("\nOnline Training completed. Final Q-table size:", len(Q_table))
for st, actions in Q_table.items():
    print("State:", st, "-> Q-values:", actions)

#plotting
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training: Cumulative Reward over Steps")
plt.legend()
plt.grid(True) 
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()