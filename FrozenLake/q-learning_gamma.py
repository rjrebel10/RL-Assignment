#Credit to Arthur Juliani
#https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np
import time
import matplotlib.pyplot as plt

def two_scales(ax1, time, data1, data2, c1, c2):
    ax2 = ax1.twinx()
    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('run')
    ax1.set_ylabel('moving average score')
    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('moving average runtime')
    return ax1, ax2

def color_y_axis(ax, color):
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None

def run_policy(env, policy, gamma = 1.0, render = False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_policy(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def Qlearning(env, alpha = 1.0, gamma = 1.0, init=0, num=10):
    if init==0:
        Q = np.zeros([env.observation_space.n,env.action_space.n])
        num_runs = num * env.observation_space.n 
    else:
        Q = np.ones([env.observation_space.n,env.action_space.n])
        num_runs = num * env.observation_space.n 
    visits = np.zeros(env.observation_space.n)
    num_observations = 1000
    rList = []
    tList = []
    for i in range(num_runs):
        start=time.time()
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < num_observations:
            j+=1
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            s1,r,d,_ = env.step(a)
            Q[s,a] = (1-alpha)*Q[s,a] + alpha*(r + gamma*np.max(Q[s1,:]))
            rAll += r
            s = s1
            visits[s] += 1
            if d == True:
                break
        rList.append(rAll)
        tList.append(time.time()-start)
    return Q,visits,rList,tList

if __name__ == '__main__':
    alpha = .8
    gamma = .8
    num=1000
    directions = ['<','V','>','^']
    envs=['FrozenLake-v0','FrozenLake8x8-v0']
    inits=[0]#,1]
    gammas=[.5,.75,.9]
    colors=['r','g','b']
    for env_name in envs:
        i=0
        for gamma in gammas:
            env = gym.make(env_name)
            Q,visits,rList,tList = Qlearning(env,alpha,gamma,0,num)
            Actions = np.array([directions[np.argmax(i)] for i in Q])
            dims = int(np.sqrt(env.observation_space.n))
            com_rList=np.convolve(np.array(rList), np.ones((num,))/num, mode='valid')
            run_List=np.arange(len(com_rList))
            plt.plot(run_List, com_rList, color=colors[i], linewidth=4, label=str(gamma))
            #np.savetxt("csvs/"+env_name+"Q-L_Alpha_"+str(int(alpha*100))+".csv", com_rList, delimiter=",")
            i+=1
        plt.legend()
        plt.title('Q Learning: '+env_name+' - Varied Gamma')
        plt.xlabel('Runs')
        plt.ylabel('Moving Average Score')
        plt.show()