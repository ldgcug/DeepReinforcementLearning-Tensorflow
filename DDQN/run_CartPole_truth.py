#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ddqn_truth import DDQN  


env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):

	natural_DQN = DDQN(s_dim = env.observation_space.shape[0],
			 a_dim = env.action_space.n,
			 learning_rate = 0.01,
			 e_greedy = 0.9,
			 replace_target_iter = 100,
			 memory_size = 2000,
			 e_greedy_increment = 0.001,
			 double_q = False,
			 sess = sess)

with tf.variable_scope('Double_DQN'):

	double_DQN = DDQN(s_dim = env.observation_space.shape[0],
			 a_dim = env.action_space.n,
			 learning_rate = 0.01,
			 e_greedy = 0.9,
			 replace_target_iter = 100,
			 memory_size = 2000,
			 e_greedy_increment = 0.001,
			 double_q = True,
			 sess = sess)


sess.run(tf.global_variables_initializer())

def train(RL):
	total_steps = 0
	total_reward = []

	EPISODE = 100
	STEP = 300
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
	return RL.q


q_natural = train(natural_DQN)
q_double = train(double_DQN)

plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()
