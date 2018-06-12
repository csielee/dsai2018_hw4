import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt

gamma = 0.7
alpha = 0.5
r_choose = 0.01

# Q(s) = Q(s) + alpha * (reward + gamma*argmax(Q(s')) - Q(s))
# observation (position, velocity)
# action 0~2 (back, stop, forward)

# state table
state_number = 20
action_number = 3

needRender = False

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

states_finish = np.zeros([state_number*2,action_number])

def RL(rewardMethod):
    states = np.zeros([state_number*2,action_number])
    cur_state = 0
    i_episode = 0

    episode_size = 1000

    
    min_f_s = env._max_episode_steps

    timeepisode = list()
    finishsteps = list()
    while(i_episode < episode_size):
        observation = env.reset()
        if needRender and i_episode % 100 == 0:
            env.render()
    
        # init state
        cur_state = observationToState(observation, env.env)
    
        if i_episode % 100 == 0:
            print("start {} episode !".format(i_episode))
    
        for t in range(env._max_episode_steps):
            if needRender and i_episode % 100 == 0:
                env.render()
        
            # choose action & random choose
            tmp = random.random()
            if (tmp >= r_choose):
                action = np.argmax(states[cur_state])
            else:
                action = random.randint(0,2)
        
            # step
            observation, reward, done, info = env.step(action)
            
            R = rewardMethod(observation, reward, done, t+1)
        
            # update state
            tmp_state = observationToState(observation, env.env)
            tmp_action = np.argmax(states[tmp_state])
        
            # update value
            states[cur_state][action] = states[cur_state][action] + alpha*(R + gamma*states[tmp_state][tmp_action] - states[cur_state][action])
            cur_state = tmp_state

            if done:
                timeepisode.append(i_episode)
                finishsteps.append(t+1)
                if t+1 < env._max_episode_steps and t+1 < min_f_s:
                    # record state
                    min_f_s = t+1
                    global states_finish
                    states_finish = states.copy()
                break
    
        i_episode = i_episode + 1
    print("finish {} episode!".format(episode_size))
    
    return (timeepisode, finishsteps)

def RLTest():
    cur_state = 0
    
    observation = env.reset()
    env.render()

    # init state
    cur_state = observationToState(observation, env.env)

    for t in range(env._max_episode_steps):
        env.render()
    
        # choose action & random choose
        action = np.argmax(states_finish[cur_state])
    
        # step
        observation, reward, done, info = env.step(action)
    
        # update state
        tmp_state = observationToState(observation, env.env)
        cur_state = tmp_state

        if done:
            if t+1 < env._max_episode_steps:
                print("done!!!")
            break
    return

def originReward(observation, reward, done, step):
    return reward

def originRewardWithDone(observation, reward, done, step):
    if done and step < env._max_episode_steps:
        return 5
    else:
        return reward
    
def calEngery(ob):
    p = ob[0]
    v = ob[1]
    return (math.sin(3 * p)*.45+.55) + v * v

def customRewardwow(ob):
    #e = calEngery(ob)
    #maxe = calEngery((0.527,0.07))
    #mine = calEngery((-0.527,0))
    #tmp = (maxe + mine)/2
    #e -= tmp
    #e *= 2
    maxr = 1
    minr = -1
    maxp = 0.6
    minp = -1.2
    p = ob[0]
    e = (p - minp)/(maxp-minp)
    e = (e * (maxr - minr)) + minr
    
    #return e*100 + p
    return e
    
    
def customReward(observation, reward, done, step):
    return customRewardwow(observation)

def customRewardWithDone(observation, reward, done, step):
    if done and step < env._max_episode_steps:
        return 5
    else:
        return customRewardwow(observation)

print("learning start")
(x,y) = RL(customRewardWithDone)

while(1):
    ret = input("test?(y/n)")
    if ret == 'n':
        break
    RLTest()
    

env.close()
