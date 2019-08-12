#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000 # renders environment if total episode reward is greater then this threshold
RENDER = False # rendering wastes time

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
		s_dim = env.observation_space.shape[0],
		a_dim = env.action_space.n,
		learning_rate = 0.02,
		reward_decay = 0.995,
		#output_graph = True
	)

for i_epsiode in range(1000):

	s = env.reset()
	while True:
		if RENDER: env.render()

		a = RL.choose_action(s)
		s_,r,done,info = env.step(a)

		RL.store_transition(s,a,r)

		if done:
			ep_rs_sum = sum(RL.ep_rs)

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

			if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
			print('episode:',i_epsiode,"reward:",int(running_reward))

			vt = RL.learn()

			if i_epsiode == 30:
				plt.plot(vt)
				plt.xlabel('episode steps')
				plt.ylabel('normalized state-action value')
				plt.show()
			break

		s = s_
