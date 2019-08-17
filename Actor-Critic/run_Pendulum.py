#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
import tensorflow as tf
from actor_critic_gaussian import Actor,Critic

env = gym.make('Pendulum-v0')
env.seed(1)     
env = env.unwrapped

N_S = env.observation_space.shape[0]
A_BOUND = env.action_space.high

DISPLAY_REWARD_THRESHOLD = -100 # renders environment if total episode reward is greater then this threshold
RENDER = False # rendering wastes time

sess = tf.Session()

actor = Actor(s_dim=N_S,action_bound=[-A_BOUND,A_BOUND],learning_rate=0.001,sess=sess)
critic = Critic(s_dim=N_S,learning_rate=0.01,reward_decay=0.9,sess=sess)

sess.run(tf.global_variables_initializer())

for i_episode in range(1000):

	s = env.reset()
	t = 0
	ep_rs = []

	while True:
		if RENDER:env.render()

		a = actor.choose_action(s)
		s_,r,done,info = env.step(a)
		r /= 10

		td_error = critic.learn(s,r,s_)
		actor.learn(s,a,td_error)

		s = s_
		t += 1
		ep_rs.append(r)

		if  t > 200:
			ep_rs_sum = sum(ep_rs)

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.9 + ep_rs_sum*0.1

			if running_reward > DISPLAY_REWARD_THRESHOLD:RENDER=True
			print('episode:',i_episode,'reward:',int(running_reward))
			break
