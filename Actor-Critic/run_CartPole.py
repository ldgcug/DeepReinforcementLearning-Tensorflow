#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
import tensorflow as tf
from actor_critic import Actor,Critic

env = gym.make('CartPole-v0')
env.seed(1)     
env = env.unwrapped

N_S = env.observation_space.shape[0]
N_A = env.action_space.n 

DISPLAY_REWARD_THRESHOLD = 400 # renders environment if total episode reward is greater then this threshold
RENDER = False # rendering wastes time

sess = tf.Session()

actor = Actor(s_dim=N_S,a_dim=N_A,learning_rate=0.01,sess=sess)
critic = Critic(s_dim=N_S,learning_rate=0.05,reward_decay=0.9,sess=sess)

sess.run(tf.global_variables_initializer())

for i_episode in range(3000):

	s = env.reset()
	t = 0
	track_r = []

	while True:
		if RENDER:env.render()

		a = actor.choose_action(s)
		s_,r,done,info = env.step(a)

		if done: r = -20
		track_r.append(r)

		td_error = critic.learn(s,r,s_)
		actor.learn(s,a,td_error)

		s = s_
		t += 1

		if done or t >= 1000:
			ep_rs_sum = sum(track_r)

			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.95 + ep_rs_sum*0.05

			if running_reward > DISPLAY_REWARD_THRESHOLD:RENDER=True
			print('episode:',i_episode,'reward:',int(running_reward))
			break
