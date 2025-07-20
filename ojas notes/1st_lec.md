# TrafIQ Self Notes and Resources

Personal Notes:

# Course 1: RL Playlist by David Silver
Link: [https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

## **Lecture-1: Introduction to RL**

1. What is the Reinforcement Learning Paradigm?
- only a reward signal, no supervisor. delayed feedback. sequential decision making
- around 2015, RL primarily used for games (attari, board etc.), humanoid robots, finance.
- Reward: Rt: **scalar** feedback signal (think why its scalar)
reward indicates an agents performance at any step ‘t’.
reward hypothesis (goal of rl) : maximization of expected cumulative reward (total future)
eg: power station: + ve reward for for producing power, -ve reward for exceeding safety thresholds (personal note: look up for exploitation and exploration in this eg)
- Sequential Decision Making: select actions which satisfy end goal
actions may have long term consequences. delayed reward. sacrifice immediate rewards.
- History: sequence of observations, actions, rewards till time ‘t’ (o, r, a are observable variables)
an agents actions depends on its history, env selects obsvn/rewards
- State: info used to determine what happens next. its a function of history. St = f(Ht)
Markov Property (for info state): future is independent of the past given the present
P[St+1 | St] = P[St+1 | S1, S2,..St]. env state is Markov state.
environment state: complete set of numbers/info of an agents surroundings at an instant
it contains everything regardless of whether the agent can perceive it or not
agent state: set of numbers/info representing agents internal model of env at an instant
its the info an agent gathers from its obsvn and uses it to pick the next action, info used by RL algos. any function of history. (personal note: understand agent and env state with egs)
proj note: list down a set of sequences for the state and proceed with experimenting them
- Fully Observable Environments: agent state = env state = info state. ideally a MDP problem.
Partially Observable Env: agent indirectly observes env. poker observes playing cards. agent≠env.
to solve a POMDP: RNN/beliefs of env state method (look these up)

1. Inside an RL Agent:
- essential components of a RL agent: model, policy, value function.
a) Policy: agents behaviour. maps from state to action. 
deterministic policy: function summarizing a decision from a state to an action. a = **π**(s)
stochastic policy: to make random decisions (conditional prob). **π(a|s) = P[A=a | S=s]**
b) Value Function: prediction of future reward. evaluates quality (good/bad) of a state. select betn actions. note: refer a couple of egs on value functions in different cases.
c) Model: predicts what the env does next. includes transitions and rewards. 
transitions predict the next state(dynamics). rewards predict the immediate reward. 
(personal note: couldn’t understand this topic properly)
- based on these 3 components: taxonomy of RL agents: (no hard and fast taxonomy, just listing)
a) value based: no implicit policy, value funcn only. 
b) policy based: no value funcn, policy only.
c) actor-critic: contains both a policy and a value function.
d) model-free: policy and/or value funcn. no model.
e) model-based: policy and/or value funcn. has a model.
1. Problems in RL: 2 fundamental probs in sqm (sequential decision making)
a) RL: env is unknown at the start. as agent interacts with the env, it improves its policy. 
b) Planning: model of env is unknown. agent improves policy by performing computations. 
exploration-exploitation trailer (classic RL thing). learning the optimal policy, not to get stuck on sub-optimal ones.
Prediction: evaluate the future, given a policy. Control: optimize the future, finding best policy.