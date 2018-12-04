#Credit to Moustafa Alzantot (malzantot@ucla.edu)
#https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

import numpy as np
import gym
from gym import wrappers
import time
import matplotlib.pyplot as plt

def run_episode(env, policy, gamma = 1.0, render = False):
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
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v

def policy_iteration(env, gamma = 1.0, interval=1):
    policy = np.random.choice(env.nA, size=(env.nS))
    max_iterations = 200000
    times = []
    scores = []
    tot_time=0
    j=0
    for i in range(max_iterations):
        j+=1
        start=time.time()
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        tot_time+=time.time()-start
        if j == interval:
            times.append(tot_time)
            scores.append(evaluate_policy(env,new_policy,1))
            j=0
        if (np.all(policy == new_policy)):
            times.append(tot_time)
            scores.append(evaluate_policy(env,new_policy,1))
            break
        policy = new_policy
    return policy,times,scores

if __name__ == '__main__':
    interval=1
    directions = ['<','V','>','^']
    envs=['FrozenLake-v0','FrozenLake8x8-v0']
    for env_name in envs:
        env = gym.make(env_name)
        env = env.unwrapped
        dims = int(np.sqrt(env.observation_space.n))
        gamma=.95
        #env.render()
        optimal_policy,times,scores = policy_iteration(env, gamma = gamma, interval=interval)
        plt.plot(times,scores)
        plt.title('Policy Iteration: '+env_name+' - Every '+str(interval)+' Iteration(s)')
        plt.xlabel('Runtime')
        plt.ylabel('Average Score')
        plt.ylim([0,1.01])
        plt.show()
        optimal_policy = np.array([directions[int(i)] for i in optimal_policy])
        print(np.reshape(optimal_policy, (dims,dims)))