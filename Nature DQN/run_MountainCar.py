#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
from dqn import DQN  

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# RL = DQN(s_dim = env.observation_space.shape[0],
# 		 a_dim = env.action_space.n,
# 		 learning_rate = 0.01,
# 		 e_greedy = 0.9,
# 		 replace_target_iter = 100,
# 		 memory_size = 2000,
# 		 e_greedy_increment = 0.001)

RL = DQN(s_dim =  env.observation_space.shape[0],
		 a_dim =  env.action_space.n,
		 learning_rate = 0.001,
		 e_greedy = 0.9,
		 replace_target_iter = 300,
		 memory_size = 3000,
		 e_greedy_increment = 0.0002)

total_steps = 0
total_reward = []
for i_episode in range(20):

	s = env.reset()
	ep_r = 0
	while True:
		env.render()
		a = RL.choose_action(s)
		s_,r,done,info = env.step(a)

		position,velocity = s_

		r = abs(position - (-0.5)) # r in [0,1]

		RL.store_transition(s,a,r,s_,done)


		if total_steps > 1000:
			RL.learn()

		ep_r += r 

		if done:
			total_reward.append(ep_r)
			get = '| Get' if s_[0] >= env.unwrapped.goal_position else '|----'
			print('Episode:',i_episode,get,'| Ep_r:',round(ep_r,4),'| Epsilon:',round(RL.epsilon,2))
			break

		s = s_
		total_steps += 1

RL.plot_cost()

import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(len(total_reward)),total_reward)
plt.ylabel('Total Reward')
plt.xlabel('Episode ')
plt.show()
