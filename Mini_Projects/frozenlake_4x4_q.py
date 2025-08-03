import gymnasium as gym 
import numpy as np

env = gym.make('FrozenLake-v1', desc = None, map_name = "4x4", is_slippery = False, render_mode = None)

alpha = 0.8 
gamma = 0.95 
epsilon = 0.95
min_epsilon = 0.01
decay_factor = 0.995
episodes = 50000
test_episodes = 1000

states = env.observation_space.n
actions = env.action_space.n
Q = np.zeros((states,actions))

for episode in range(episodes):
    state, _ = env.reset()
    terminated = False #true when elf falls in a hole or reaches goal
    truncated = False #true when elf actions > 100

    while not (truncated or terminated):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        new_state, reward, terminated, truncated, _ = env.step(action)

        if terminated and reward != 1.0:
            reward = -1
        elif new_state == state:
            reward = - 0.01

        #q-learning eqn: Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
        optimal_action = np.argmax(Q[new_state])
        Q[state, action] += alpha * (reward + gamma * Q[new_state, optimal_action] - Q[state, action])
        state = new_state

    epsilon = max(min_epsilon, epsilon * decay_factor)
print("Training complete")
np.save("frozenlake_4x4_qtable.npy", Q)

def run(n):
    success = 0
    print(f"\nTesting over {n} episodes:")
    for episode in range(n):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        print(f"\nEpisode{episode}:")

        while not (terminated or truncated):
            action = Q[state].argmax()
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = new_state

        if terminated: 
            if reward == 1:
                print("Yay!! You reached the gift")
                success += 1
            else:
                print("Oops! Try again!")

        print(f"Final Reward: {total_reward}")
        print(f"\nReached the goal in {success} out of {n} episodes")
    env.close()    

if __name__ == '__main__':
    run(test_episodes)