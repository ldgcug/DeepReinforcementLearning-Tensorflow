#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math
import random
import numpy as np  
import tensorflow as tf 
from collections import deque  

# np.random.seed(1)
# tf.set_random_seed(1)

class DueilingDQN(object):
	def __init__(self,
		s_dim,
		a_dim,
		learning_rate = 0.01,
		reward_decay = 0.9,
		e_greedy = 0.9,
		replace_target_iter = 200,
		memory_size = 500,
		batch_size = 32,
		e_greedy_increment = None,
		output_graph = False,
		dueling = True,
		sess = None
	):
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		#create memory 
		self.memory = deque()
		self.memory_count = 0
		self.memory_size = memory_size
		self.batch_size = batch_size

		self.dueling = dueling

		#toal learning step
		self.learn_step_counter = 0

		#consist of [target_net,evaluate_net]
		self.build_net()

		if sess is None:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
		else:
			self.sess = sess

		self.cost_his = []

	def build_net(self):
		# ************************ build evaluate net *****************************
		self.s = tf.placeholder(tf.float32,[None,self.s_dim],name='s_dim') #input
		self.q_target = tf.placeholder(tf.float32,[None,self.a_dim],name='q_target') #for calculating loss

		w_initializer = tf.random_normal_initializer(0.,0.3)
		b_initializer = tf.constant_initializer(0.1)

		with tf.variable_scope('eval_net'):
			c_names = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
			
			with tf.variable_scope('layer1'):
				w1 = tf.get_variable('w1',[self.s_dim,20],initializer=w_initializer,collections=c_names)
				b1 = tf.get_variable('b1',[20],initializer=b_initializer,collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)


			if self.dueling:
				#Dueling DQN
				with tf.variable_scope('Value'):
					w2 = tf.get_variable('w2',[20,1],initializer=w_initializer,collections=c_names)
					b2 = tf.get_variable('b2',[1],initializer=b_initializer,collections=c_names)
					self.V = tf.matmul(l1,w2) + b2

				with tf.variable_scope('Advantage'):
					w2 = tf.get_variable('w2',[20,self.a_dim],initializer=w_initializer,collections=c_names)
					b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,collections=c_names)
					self.A = tf.matmul(l1,w2) + b2

				with tf.variable_scope('Q'):
					self.q_eval = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True))
			else:
				with tf.variable_scope('Q'):
					w2 = tf.get_variable('w2',[20,self.a_dim],initializer=w_initializer,collections=c_names)
					b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,collections=c_names)
					self.q_eval = tf.matmul(l1,w2) + b2
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))

		with tf.variable_scope('train'):
			self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

		# ************************ build target net *****************************
		self.s_ = tf.placeholder(tf.float32,[None,self.s_dim],name='s_') #input
		with tf.variable_scope('target_net'):
			c_names = ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]

			with tf.variable_scope('layer1'):
				w1 = tf.get_variable('w1',[self.s_dim,20],initializer=w_initializer,collections=c_names)
				b1 = tf.get_variable('b1',[20],initializer=b_initializer,collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s_,w1)+b1)

			if self.dueling:
				#Dueling DQN
				with tf.variable_scope('Value'):
					w2 = tf.get_variable('w2',[20,1],initializer=w_initializer,collections=c_names)
					b2 = tf.get_variable('b2',[1],initializer=b_initializer,collections=c_names)
					self.V = tf.matmul(l1,w2) + b2

				with tf.variable_scope('Advantage'):
					w2 = tf.get_variable('w2',[20,self.a_dim],initializer=w_initializer,collections=c_names)
					b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,collections=c_names)
					self.A = tf.matmul(l1,w2) + b2

				with tf.variable_scope('Q'):
					self.q_next = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True))

			else:
				with tf.variable_scope('Q'):
					w2 = tf.get_variable('w2',[20,self.a_dim],initializer=w_initializer,collections=c_names)
					b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,collections=c_names)
					self.q_next = tf.matmul(l1,w2) + b2
	
	def store_transition(self,s,a,r,s_,done):
		transition = (s,a,r,s_,done)
		if self.memory_count < self.memory_size:
			self.memory.append(transition)
			self.memory_count += 1
		else:
			self.memory.popleft()
			self.memory.append(transition)


	def choose_action(self,state):

		if np.random.uniform() < self.epsilon:
			return np.argmax(self.sess.run(self.q_eval,feed_dict={self.s:state.reshape(-1,self.s_dim)}))

		return np.random.randint(0,self.a_dim)

	def learn(self):
		# cheak ro replace target parameters
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.train_target()
			print('\n target_params_replaces \n')

		# sample batch memory from all memory
		if self.memory_count > self.batch_size:
			mini_batch = random.sample(self.memory,self.batch_size)
		else:
			mini_batch = random.sample(self.memory,self.memory_count)

		states = np.asarray([e[0] for e in mini_batch])
		actions = np.asarray([e[1] for e in mini_batch])
		rewards = np.asarray([e[2] for e in mini_batch])
		next_states = np.asarray([e[3] for e in mini_batch])
		dones = np.asarray([e[4] for e in mini_batch])

		q_next = self.sess.run(self.q_next,feed_dict={self.s_:next_states})
		q_eval = self.sess.run(self.q_eval,feed_dict={self.s:states})
		q_target = q_eval.copy()

		for k in range(len(mini_batch)):
			if dones[k]:
				q_target[k][actions[k]] = rewards[k]
			else:
				q_target[k][actions[k]]	= rewards[k] + self.gamma * np.max(q_next[k])

		loss = self.sess.run(self.loss,feed_dict={self.s:states,self.q_target:q_target})
		self.cost_his.append(loss)
		_ = self.sess.run(self.train_op,feed_dict={self.s:states,self.q_target:q_target})

		# increasing epsilon
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1

	def train_target(self):
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])

	def plot_cost(self):
		import matplotlib.pyplot as plt
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()