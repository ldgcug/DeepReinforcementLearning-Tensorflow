#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
from ddqn import DDQN  


env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DDQN(s_dim = env.observation_space.shape[0],
		 a_dim = env.action_space.n,
		 learning_rate = 0.01,
		 e_greedy = 0.9,
		 replace_target_iter = 100,
		 memory_size = 2000,
		 e_greedy_increment = 0.001)

total_steps = 0
total_reward = []

EPISODE = 100
STEP = 300
TEST = 5

for i_episode in range(EPISODE):
	s = env.reset()
	ep_r = 0
	for step in range(STEP):
		env.render()

		a = RL.choose_action(s)
		s_,r,done,info = env.step(a)
		# the smaller theta and closer to center the better
		x, x_dot, theta, theta_dot = s_
		r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
		r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
		r = r1 + r2
		RL.store_transition(s,a,r,s_,done)

		ep_r += r 


		if total_steps > 1000:
			RL.learn()

		if done:
			print('episode:',i_episode,'ep_r:',round(ep_r,2),'epsilon',round(RL.epsilon,2),'buffer_size:',RL.memory_count,'steps:',total_steps)
			total_reward.append(ep_r)
			break

		s = s_
		total_steps += 1

	# Test every 100 episodes
	if i_episode % 10 == 0:
		total_rewards = 0
		for i in range(TEST):
			state = env.reset()
			for j in range(STEP):
				env.render()
				action = RL.get_action(state) # direct action for test
				state,reward,done,_ = env.step(action)
				total_rewards += reward
				if done:
					break
		ave_reward = total_rewards/TEST
		print ('episode: ',i_episode,'Evaluation Average Reward:',ave_reward)

RL.plot_cost()

import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(len(total_reward)),total_reward)
plt.ylabel('Total Reward')
plt.xlabel('Episode ')
plt.show()
