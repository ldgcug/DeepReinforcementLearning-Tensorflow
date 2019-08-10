#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dueling_dqn import DueilingDQN 


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

MEMORY_SIZE = 3000
ACTION_SPACE = 15

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):

	natural_DQN = DueilingDQN(s_dim = env.observation_space.shape[0],
             a_dim = ACTION_SPACE,
             learning_rate = 0.001,
             memory_size = MEMORY_SIZE,
             e_greedy_increment = 0.001,
             dueling = False,
             sess = sess)

with tf.variable_scope('Duelin_DQN'):

	dueling_DQN = DueilingDQN(s_dim = env.observation_space.shape[0],
             a_dim = ACTION_SPACE,
             learning_rate = 0.001,
             memory_size = MEMORY_SIZE,
             e_greedy_increment = 0.001,
             dueling = True,
             sess = sess)



sess.run(tf.global_variables_initializer())

def train(RL):
	acc_r = [0]
	total_steps = 0
	s = env.reset()

	while True:
		# env.render()
		a = RL.choose_action(s)
		f_action = (a-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
		s_, r, done, info = env.step(np.array([f_action]))

		r /= 10 # normalize to a range of (-1, 0). r = 0 when get upright
		acc_r.append(r + acc_r[-1]) # accumulated reward
		RL.store_transition(s,a,r,s_,done)

		if total_steps % 1000 ==0 :
			print('current steps:',total_steps)

		if total_steps > MEMORY_SIZE: #learning
			RL.learn()

		if total_steps - MEMORY_SIZE > 15000: # stop game
			break

		s = s_
		total_steps += 1

	return RL.cost_his,acc_r

c_natural,r_natural = train(natural_DQN)
c_dueling,r_dueling = train(dueling_DQN)

plt.plot(np.array(c_natural), c='r', label='natural')
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()

plt.figure(2)
plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.show()


