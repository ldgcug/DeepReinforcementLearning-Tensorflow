#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from prioritized import DQNPrioritizedReplay  


env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(21)

MEMORY_SIZE = 10000

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):

	natural_DQN = DQNPrioritizedReplay(s_dim = env.observation_space.shape[0],
			 a_dim = env.action_space.n,
			 memory_size = MEMORY_SIZE,
			 e_greedy_increment = 0.00005,
			 prioritized = False,
			 sess = sess)

with tf.variable_scope('DQN_with_prioritized_replay'):

	poritized_DQN = DQNPrioritizedReplay(s_dim = env.observation_space.shape[0],
			 a_dim = env.action_space.n,
			 memory_size = MEMORY_SIZE,
			 e_greedy_increment = 0.00005,
			 prioritized = True,
			 sess = sess)


sess.run(tf.global_variables_initializer())

def train(RL):
	total_steps = 0
	steps = []
	episodes = []

	EPISODE = 20
	for i_episode in range(EPISODE):
		s = env.reset()
		while True:
			#env.render()

			a = RL.choose_action(s)
			s_,r,done,info = env.step(a)
			
			if done:
				r = 10

			RL.store_transition(s,a,r,s_,done)

			if total_steps % 1000 == 0:
				print('current steps:',total_steps)


			if total_steps > MEMORY_SIZE:
				RL.learn()

			if done:
				print('episode:',i_episode,'finished')
				steps.append(total_steps)
				episodes.append(i_episode)
				break

			s = s_
			total_steps += 1
	return np.vstack((episodes, steps))


q_natural = train(natural_DQN)
q_prioritized= train(poritized_DQN)
#q_natural = train(natural_DQN)



# compare based on first success
plt.plot(q_natural[0, :], q_natural[1, :] - q_natural[1, 0], c='b', label='natural DQN')
plt.plot(q_prioritized[0, :], q_prioritized[1, :] - q_prioritized[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()
