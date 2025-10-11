import gymnasium as g
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

def run(e,istraining=True,render=False):
    world=g.make('Taxi-v3',render_mode='human'if render else None)

    if istraining==True:
        state_size=world.observation_space.n
        action_size=world.action_space.n
        q_table=np.zeros((state_size,action_size))
    else:
        f=open('taxi.pkl','rb')
        q_table= pickle.load(f)
        f.close()

    learning_rate=0.9
    discount_rate=0.8
    epsilon=1.0
    decay_rate=0.000125
    action_count=0
    num_episodes=e
    max_steps=200
    successrate=0

    r=np.zeros(e)
    for i in range(num_episodes):
        state=world.reset()[0]
        done=False #done is an indicator of whether the passenger reached dest
        rewards=0 #total rewards per episode
        for s in range(max_steps):
            if istraining and random.uniform(0,1)<epsilon:
                action=world.action_space.sample()
            else:
                action=np.argmax(q_table[state,:])
            new_state,reward,terminated,truncated,_=world.step(action)
            action_count+=1
            done=terminated or truncated
            rewards+=reward

            if istraining==True:
                q_table[state,action]=q_table[state,action]+learning_rate*(reward+discount_rate*np.max(q_table[new_state,:])-q_table[state,action])
            state=new_state
            if done==True:
                if terminated:
                    successrate+=1
                break
        epsilon=max(epsilon-decay_rate,0)
        if(epsilon==0):
            learning_rate = 0.0001
        r[i]=rewards
    print("Action count: ",action_count)
    if not(istraining):
        print("Success Rate: ",successrate,"/",e)
    world.close()

    sum_rewards = np.zeros(e)
    for t in range(e):
        sum_rewards[t] = np.sum(r[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')

    if istraining:
        f = open("taxi.pkl","wb")
        pickle.dump(q_table, f)
        f.close()
#run(9000,True,False) #Uncomment for training model
run(10,False,True) #Uncomment for testing
print("done")
