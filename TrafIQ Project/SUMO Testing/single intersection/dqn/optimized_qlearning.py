import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# Enhanced SUMO configuration
Sumo_config = [
    'sumo',  # Remove GUI for faster training
    '-c', 'dqn.sumocfg',
    '--step-length', '1.0',
    '--no-step-log', 'true',
    '--no-warnings', 'true',
    '--waiting-time-memory', '10000',
    '--time-to-teleport', '-1'
]

# Improved hyperparameters
TOTAL_EPISODES = 1500
MAX_STEPS_PER_EPISODE = 1800  # 30 minutes simulation time
ALPHA = 0.1  # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05

# Enhanced action space
MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 90
GREEN_TIME_STEP = 5
GREEN_TIMES = list(range(MIN_GREEN_TIME, MAX_GREEN_TIME + 1, GREEN_TIME_STEP))

# Directions: 0=NS, 1=EW
DIRECTIONS = ['NS', 'EW']
ACTIONS = []
for direction in range(len(DIRECTIONS)):
    for green_time_idx in range(len(GREEN_TIMES)):
        ACTIONS.append((direction, green_time_idx))

print(f"Total actions: {len(ACTIONS)}")

# State discretization parameters
QUEUE_BINS = [0, 3, 7, 12, 20, float('inf')]  # More granular binning
WAITING_BINS = [0, 30, 60, 120, 300, float('inf')]
SPEED_BINS = [0, 10, 25, 40, float('inf')]
TIME_BINS = [0, 20, 40, 80, float('inf')]

class EnhancedQLearning:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
        self.state_visits = defaultdict(int)
        self.episode_rewards = []
        self.episode_metrics = []
        
        # Performance tracking
        self.metrics_history = {
            'queue_length': [],
            'waiting_time': [],
            'co2_emissions': [],
            'throughput': [],
            'avg_speed': []
        }
        
        # Traffic light state
        self.current_phase = 0  # 0=NS, 1=EW
        self.last_switch_step = 0
        self.current_step = 0
        
        # Previous metrics for reward calculation
        self.prev_metrics = {}

    def discretize_value(self, value, bins):
        """Discretize continuous values into bins"""
        for i, threshold in enumerate(bins[:-1]):
            if value <= threshold:
                return i
        return len(bins) - 2

    def get_comprehensive_metrics(self):
        """Get comprehensive traffic metrics"""
        vehicle_ids = traci.vehicle.getIDList()
        
        if not vehicle_ids:
            return {
                'total_vehicles': 0,
                'avg_waiting_time': 0,
                'avg_speed': 0,
                'total_co2': 0,
                'total_fuel': 0,
                'queue_lengths': [0, 0, 0, 0],
                'total_queue': 0
            }
        
        # Calculate metrics
        total_waiting = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
        total_speed = sum(traci.vehicle.getSpeed(vid) for vid in vehicle_ids)
        total_co2 = sum(traci.vehicle.getCO2Emission(vid) for vid in vehicle_ids)
        total_fuel = sum(traci.vehicle.getFuelConsumption(vid) for vid in vehicle_ids)
        
        # Queue lengths
        q_EB = self.get_queue_length(["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2"])
        q_SB = self.get_queue_length(["Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"])
        q_WB = self.get_queue_length(["Node2_3_WB_0", "Node2_3_WB_1", "Node2_3_WB_2"])
        q_NB = self.get_queue_length(["Node2_5_NB_0", "Node2_5_NB_1", "Node2_5_NB_2"])
        
        queue_lengths = [q_EB, q_SB, q_WB, q_NB]
        
        return {
            'total_vehicles': len(vehicle_ids),
            'avg_waiting_time': total_waiting / len(vehicle_ids),
            'avg_speed': total_speed / len(vehicle_ids),
            'total_co2': total_co2,
            'total_fuel': total_fuel,
            'queue_lengths': queue_lengths,
            'total_queue': sum(queue_lengths)
        }

    def get_queue_length(self, lane_ids):
        """Get total halting vehicles in specified lanes"""
        return sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids)

    def get_state(self):
        """Enhanced state representation with better discretization"""
        metrics = self.get_comprehensive_metrics()
        
        # Discretize queue lengths for each direction
        queue_states = []
        for queue in metrics['queue_lengths']:
            queue_states.append(self.discretize_value(queue, QUEUE_BINS))
        
        # Discretize other important metrics
        waiting_state = self.discretize_value(metrics['avg_waiting_time'], WAITING_BINS)
        speed_state = self.discretize_value(metrics['avg_speed'], SPEED_BINS)
        
        # Time since last phase switch
        time_since_switch = self.current_step - self.last_switch_step
        time_state = self.discretize_value(time_since_switch, TIME_BINS)
        
        # Current phase
        phase_state = self.current_phase
        
        # Construct state tuple
        state = tuple(queue_states + [waiting_state, speed_state, time_state, phase_state])
        
        return state

    def get_multi_objective_reward(self):
        """Enhanced reward function considering multiple objectives"""
        current_metrics = self.get_comprehensive_metrics()
        
        if not self.prev_metrics:
            self.prev_metrics = current_metrics
            return 0.0
        
        # Reward weights for different objectives
        weights = {
            'waiting_time': -1.5,
            'queue_length': -1.0,
            'co2_emissions': -0.8,
            'fuel_consumption': -0.5,
            'speed': 1.0,
            'balance': -0.5  # Penalty for imbalanced queues
        }
        
        # Calculate improvements
        waiting_improvement = self.prev_metrics['avg_waiting_time'] - current_metrics['avg_waiting_time']
        queue_improvement = self.prev_metrics['total_queue'] - current_metrics['total_queue']
        co2_improvement = self.prev_metrics['total_co2'] - current_metrics['total_co2']
        fuel_improvement = self.prev_metrics['total_fuel'] - current_metrics['total_fuel']
        speed_improvement = current_metrics['avg_speed'] - self.prev_metrics['avg_speed']
        
        # Queue balance penalty (prefer balanced queues)
        queue_std = np.std(current_metrics['queue_lengths'])
        balance_penalty = queue_std
        
        # Calculate weighted reward
        reward = (weights['waiting_time'] * waiting_improvement +
                  weights['queue_length'] * queue_improvement +
                  weights['co2_emissions'] * co2_improvement / 1000.0 +
                  weights['fuel_consumption'] * fuel_improvement / 1000.0 +
                  weights['speed'] * speed_improvement +
                  weights['balance'] * balance_penalty)
        
        # Additional penalties and bonuses
        
        # Severe penalty for very long queues
        max_queue = max(current_metrics['queue_lengths'])
        if max_queue > 20:
            reward -= (max_queue - 20) * 0.8
        
        # Bonus for maintaining good flow
        if current_metrics['avg_speed'] > 20 and current_metrics['total_queue'] < 15:
            reward += 1.0
        
        # Penalty for excessive waiting
        if current_metrics['avg_waiting_time'] > 120:
            reward -= 2.0
        
        self.prev_metrics = current_metrics
        return reward

    def get_action(self, state):
        """Epsilon-greedy action selection with improved exploration"""
        self.state_visits[state] += 1
        
        # Adaptive epsilon based on state visits (UCB-style exploration)
        visit_bonus = 1.0 / (1.0 + self.state_visits[state])
        effective_epsilon = max(EPSILON_MIN, EPSILON + visit_bonus * 0.1)
        
        if random.random() < effective_epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        else:
            return int(np.argmax(self.q_table[state]))

    def apply_action(self, action_idx, tls_id="Node2"):
        """Apply action with proper phase management"""
        direction, green_time_idx = ACTIONS[action_idx]
        green_time = GREEN_TIMES[green_time_idx]
        
        # Phase mapping
        target_phase = 0 if direction == 0 else 2  # NS=0, EW=2
        
        # Check if phase change is needed and minimum time has elapsed
        min_phase_time = 15
        if (target_phase != self.current_phase and 
            (self.current_step - self.last_switch_step) >= min_phase_time):
            
            # Yellow phase transition
            yellow_phase = self.current_phase + 1
            traci.trafficlight.setPhase(tls_id, yellow_phase)
            
            # Apply yellow for 3 seconds
            for _ in range(3):
                if traci.simulation.getMinExpectedNumber() > 0:
                    traci.simulationStep()
                    self.current_step += 1
            
            self.current_phase = target_phase
            self.last_switch_step = self.current_step
        
        # Apply green phase
        traci.trafficlight.setPhase(tls_id, self.current_phase)
        
        # Run green phase for specified duration
        for _ in range(green_time):
            if traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                self.current_step += 1
            else:
                break

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table with learning"""
        old_q = self.q_table[state][action]
        best_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update with adaptive learning rate
        adaptive_alpha = ALPHA / (1 + self.state_visits[state] * 0.001)
        new_q = old_q + adaptive_alpha * (reward + GAMMA * best_next_q - old_q)
        
        self.q_table[state][action] = new_q

    def train(self):
        """Enhanced training loop"""
        global EPSILON
        
        print("Starting Enhanced Q-Learning Training...")
        print(f"Episodes: {TOTAL_EPISODES}")
        print(f"Actions: {len(ACTIONS)} (Directions: {len(DIRECTIONS)}, Green times: {len(GREEN_TIMES)})")
        
        for episode in range(TOTAL_EPISODES):
            # Start SUMO
            traci.start(Sumo_config)
            
            # Reset episode variables
            self.current_phase = 0
            self.last_switch_step = 0
            self.current_step = 0
            self.prev_metrics = {}
            
            # Warm up simulation
            for _ in range(10):
                traci.simulationStep()
                self.current_step += 1
            
            episode_reward = 0.0
            episode_metrics = []
            step_count = 0
            
            state = self.get_state()
            
            while (traci.simulation.getMinExpectedNumber() > 0 and 
                   step_count < MAX_STEPS_PER_EPISODE):
                
                # Get action
                action = self.get_action(state)
                
                # Apply action
                self.apply_action(action)
                
                # Get new state and reward
                next_state = self.get_state()
                reward = self.get_multi_objective_reward()
                
                # Update Q-table
                self.update_q_table(state, action, reward, next_state)
                
                # Update tracking variables
                episode_reward += reward
                step_count += 1
                
                # Collect metrics
                current_metrics = self.get_comprehensive_metrics()
                episode_metrics.append(current_metrics)
                
                state = next_state
            
            # Episode finished
            traci.close()
            
            # Decay epsilon
            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
            
            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            
            if episode_metrics:
                avg_episode_metrics = {
                    'queue_length': np.mean([m['total_queue'] for m in episode_metrics]),
                    'waiting_time': np.mean([m['avg_waiting_time'] for m in episode_metrics]),
                    'co2_emissions': np.mean([m['total_co2'] for m in episode_metrics]),
                    'throughput': np.mean([m['total_vehicles'] for m in episode_metrics]),
                    'avg_speed': np.mean([m['avg_speed'] for m in episode_metrics])
                }
                
                for key in self.metrics_history:
                    self.metrics_history[key].append(avg_episode_metrics[key])
            
            # Progress reporting
            if (episode + 1) % 50 == 0:
                recent_reward = np.mean(self.episode_rewards[-50:])
                recent_queue = np.mean(self.metrics_history['queue_length'][-50:])
                recent_waiting = np.mean(self.metrics_history['waiting_time'][-50:])
                q_table_size = len(self.q_table)
                
                print(f"Episode {episode+1}/{TOTAL_EPISODES} | "
                      f"Avg Reward: {recent_reward:.2f} | "
                      f"Avg Queue: {recent_queue:.1f} | "
                      f"Avg Wait: {recent_waiting:.1f}s | "
                      f"Q-table size: {q_table_size} | "
                      f"Epsilon: {EPSILON:.3f}")
        
        print("\nTraining completed!")
        print(f"Final Q-table size: {len(self.q_table)}")
        
        # Save Q-table
        with open('enhanced_q_table.pkl', 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def plot_results(self):
        """Plot comprehensive training results"""
        episodes = list(range(1, len(self.episode_rewards) + 1))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Episode rewards
        axes[0,0].plot(episodes, self.episode_rewards, alpha=0.6, label="Episode Reward")
        window = 100
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode="valid")
            axes[0,0].plot(episodes[window-1:], moving_avg, label=f"Moving Avg ({window})", color="red")
        axes[0,0].set_xlabel("Episode")
        axes[0,0].set_ylabel("Reward")
        axes[0,0].set_title("Training Rewards")
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Queue length
        axes[0,1].plot(episodes, self.metrics_history['queue_length'], alpha=0.6, label="Queue Length")
        if len(self.metrics_history['queue_length']) >= window:
            moving_avg = np.convolve(self.metrics_history['queue_length'], np.ones(window)/window, mode="valid")
            axes[0,1].plot(episodes[window-1:], moving_avg, label=f"Moving Avg ({window})", color="orange")
        axes[0,1].set_xlabel("Episode")
        axes[0,1].set_ylabel("Avg Queue Length")
        axes[0,1].set_title("Queue Length Optimization")
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Waiting time
        axes[0,2].plot(episodes, self.metrics_history['waiting_time'], alpha=0.6, label="Waiting Time")
        if len(self.metrics_history['waiting_time']) >= window:
            moving_avg = np.convolve(self.metrics_history['waiting_time'], np.ones(window)/window, mode="valid")
            axes[0,2].plot(episodes[window-1:], moving_avg, label=f"Moving Avg ({window})", color="green")
        axes[0,2].set_xlabel("Episode")
        axes[0,2].set_ylabel("Avg Waiting Time (s)")
        axes[0,2].set_title("Waiting Time Optimization")
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        # CO2 emissions
        axes[1,0].plot(episodes, self.metrics_history['co2_emissions'], alpha=0.6, label="CO2 Emissions")
        if len(self.metrics_history['co2_emissions']) >= window:
            moving_avg = np.convolve(self.metrics_history['co2_emissions'], np.ones(window)/window, mode="valid")
            axes[1,0].plot(episodes[window-1:], moving_avg, label=f"Moving Avg ({window})", color="brown")
        axes[1,0].set_xlabel("Episode")
        axes[1,0].set_ylabel("Avg CO2 Emissions")
        axes[1,0].set_title("CO2 Emissions Optimization")
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Average speed
        axes[1,1].plot(episodes, self.metrics_history['avg_speed'], alpha=0.6, label="Average Speed")
        if len(self.metrics_history['avg_speed']) >= window:
            moving_avg = np.convolve(self.metrics_history['avg_speed'], np.ones(window)/window, mode="valid")
            axes[1,1].plot(episodes[window-1:], moving_avg, label=f"Moving Avg ({window})", color="purple")
        axes[1,1].set_xlabel("Episode")
        axes[1,1].set_ylabel("Average Speed (m/s)")
        axes[1,1].set_title("Speed Optimization")
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Final performance comparison
        final_metrics = [
            np.mean(self.metrics_history['queue_length'][-50:]),
            np.mean(self.metrics_history['waiting_time'][-50:]),
            np.mean(self.metrics_history['co2_emissions'][-50:]) / 1000,  # Scale for visibility
            np.mean(self.metrics_history['avg_speed'][-50:])
        ]
        metric_names = ['Queue Length', 'Waiting Time', 'CO2/1000', 'Avg Speed']
        axes[1,2].bar(metric_names, final_metrics)
        axes[1,2].set_title("Final Performance (Last 50 Episodes)")
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("enhanced_qlearning_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final performance summary
        print(f"\nFinal Performance Summary (Last 50 Episodes):")
        print(f"Average Queue Length: {final_metrics[0]:.2f}")
        print(f"Average Waiting Time: {final_metrics[1]:.2f}s")
        print(f"Average CO2 Emissions: {np.mean(self.metrics_history['co2_emissions'][-50:]):.2f}")
        print(f"Average Speed: {final_metrics[3]:.2f} m/s")
        print(f"Average Reward: {np.mean(self.episode_rewards[-50:]):.2f}")

if __name__ == "__main__":
    # Initialize and train the enhanced Q-learning agent
    agent = EnhancedQLearning()
    agent.train()
    agent.plot_results()
    
    print("\nTraining completed! Enhanced Q-table saved to 'enhanced_q_table.pkl'")
    print("Results plots saved to 'enhanced_qlearning_results.png'")