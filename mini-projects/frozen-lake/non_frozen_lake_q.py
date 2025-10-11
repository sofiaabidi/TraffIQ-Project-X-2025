import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def train_agent(episodes, render=False):
    env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=True, render_mode=None)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    alpha = 0.6
    gamma = 0.99
    epsilon = 1.0
    linear_decay_rate = 0.00005
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action]
            )
            state = new_state

        epsilon = max(epsilon - linear_decay_rate, 0)
        if epsilon == 0:
            alpha = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(sum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Moving Average of Rewards (over 100 episodes)')
    plt.title('Training Progress of Elf')
    plt.grid(True)
    plt.savefig('frozen_lake_4x4.png')
    plt.close()

    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q, f)

def test(episodes=100):
    try:
        with open("q_table.pkl", "rb") as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        print("Error! Q-table not found. Please train the agent first.")
        return

    env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=True, render_mode='human')

    epsilon = 0.01
    rng = np.random.default_rng()
    total_success = 0

    print("Testing the agent!")
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state

        if reward == 1:
            total_success += 1
            print(f"Episode {i+1}: Yay! The elf reached the gift!")
        else:
            print(f"Episode {i+1}: Oops! Try again!")

    env.close()
    success_rate = (total_success / episodes) * 100
    print(f"\nSuccess rate over {episodes} test episodes: {success_rate:.2f}%")


if __name__ == '__main__':
    train_agent(25000)
    test(100)