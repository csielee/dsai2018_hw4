import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt

gamma = 0.95
alpha = 0.5
r_choose = 0.0

# Q(s) = Q(s) + alpha * (reward + gamma*argmax(Q(s')) - Q(s))
# observation (position, velocity)
# action 0~2 (back, stop, forward)

# state table
state_number = 20
action_number = 3
states = np.zeros([state_number*2,action_number])

def observationToState(ob, e):
    position = ob[0]
    velocity = ob[1]
    if velocity >= 0:
        v = 1
    else:
        v = 0
    section = (e.max_position - e.min_position)/state_number
    p = position - e.min_position
    p = int(p/section)
    s = p + v*state_number
    return s

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000
#print(env)
cur_state = 0
i_episode = 0
train_finish = False
x = list()
y = list()
while(not train_finish):
    observation = env.reset()
    env.render()
    # init state
    cur_state = observationToState(observation, env.env)
    
    if i_episode % 20 == 0:
        #ret = input("start {} episode !".format(i_episode))
        print("start {} episode !".format(i_episode))
        plt.plot(x,y)
        plt.show()
        ret = 'c'
        if ret == 'e':
            break
    
    for t in range(env._max_episode_steps):
        env.render()
        #print(observation)
        #action = env.action_space.sample()
        #if observation[1] >= 0:
        #    action = 2
        #else:
        #    action = 0
        
        # choose action & random choose
        tmp = random.random()
        if (tmp >= r_choose):
            action = np.argmax(states[cur_state])
        else:
            action = random.randint(0,2)
        #input("action : {}".format(action))
        observation, reward, done, info = env.step(action)
        # update state
        tmp_state = observationToState(observation, env.env)
        tmp_action = np.argmax(states[tmp_state])
        #input("tmp action : {}".format(tmp_action))
        # update value
        states[cur_state][action] = states[cur_state][action] + alpha*(reward + gamma*states[tmp_state][tmp_action] - states[cur_state][action])
        cur_state = tmp_state

        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print(states)
            x.append(i_episode)
            y.append(t+1)
            #if t+1 < env._max_episode_steps:
                #input('realy finish!')
                #train_finish = True
            break
            
        
    i_episode = i_episode + 1

print("train ending!")
input("test starting!")
i_episode = 0
while(1):
    observation = env.reset()
    env.render()
    # init state
    cur_state = observationToState(observation, env.env)
    
    ret = input("start test {} episode !".format(i_episode))
    if ret == 'e':
        break
    
    for t in range(env._max_episode_steps):
        env.render()
        
        # choose action & random choose
        tmp = random.random()
        if (tmp >= r_choose):
            action = np.argmax(states[cur_state])
        else:
            action = random.randint(0,2)
        
        observation, reward, done, info = env.step(action)
        # update state
        cur_state = observationToState(observation, env.env)
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            if t+1 < env._max_episode_steps:
                print('realy finish!')
            break
    
    i_episode = i_episode + 1

env.close()
