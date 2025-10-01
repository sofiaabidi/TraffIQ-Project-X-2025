from pettingzoo.classic import tictactoe_v3
import numpy as np
import random
from collections import defaultdict

Q_tables = {
    "player_1": defaultdict(lambda: np.zeros(9)),
    "player_2": defaultdict(lambda: np.zeros(9))
}

alpha = 0.8
gamma = 0.95
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.999
episodes = 100000

def get_state(obs):
    return str(obs["observation"].tolist())

def choose_action(agent, state, action_mask):
    if random.random() < epsilon:
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        return random.choice(valid_actions)
    q_values = Q_tables[agent][state].copy()
    q_values = np.where(action_mask, q_values, -np.inf)
    return int(np.argmax(q_values))

def update_q(agent, state, action, reward, next_state, next_mask, done):
    current_q = Q_tables[agent][state][action]
    if done:
        target = reward
    else:
        future_q = Q_tables[agent][next_state].copy()
        future_q = np.where(next_mask, future_q, -np.inf)
        target = reward + gamma * np.max(future_q)
    Q_tables[agent][state][action] += alpha * (target - current_q)

# Track outcomes
win_stats = {"player_1": 0, "player_2": 0, "draws": 0}

for ep in range(episodes):
    env = tictactoe_v3.env()
    env.reset()
    last_obs = {}

    for agent in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()

        state = get_state(obs)
        mask = np.array(obs["action_mask"], dtype=bool)

        if agent in last_obs:
            prev_agent, prev_state, prev_action, prev_mask = last_obs[agent]
            update_q(prev_agent, prev_state, prev_action, reward, state, mask, terminated or truncated)

        if not terminated and not truncated:
            action = choose_action(agent, state, mask)
            last_obs[agent] = (agent, state, action, mask)
        else:
            action = None

        env.step(action)

        if terminated or truncated:
            # final rewards
            if reward == 1:
                win_stats[agent] += 1
            elif reward == 0:
                win_stats["draws"] += 1

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (ep + 1) % 1000 == 0:
        print(f"Episode {ep+1}/{episodes} | ε={epsilon:.3f} | Wins: P1={win_stats['player_1']} P2={win_stats['player_2']} Draws={win_stats['draws']}")