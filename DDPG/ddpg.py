#!/usr/bin/env python
#-*- coding: utf-8 -*-

import random
import numpy as np 
import tensorflow as tf 
from collections import deque


class DDPG(object):
	def __init__(self,s_dim,a_dim,a_bound):
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.a_bound = a_bound

		self.gamma = 0.9
		self.tau = 0.01
		self.lr_c = 0.002
		self.lr_a = 0.001

		self.memory = deque()
		self.memory_size = 10000
		self.memory_count = 0
		self.batch_size = 32

		self.sess = tf.Session()

		self.s = tf.placeholder(tf.float32,[None,self.s_dim],name='s')
		self.s_ = tf.placeholder(tf.float32,[None,self.s_dim],name='s_')
		self.r = tf.placeholder(tf.float32,[None,1],name='r')

		with tf.variable_scope('Actor'):
			self.a = self.build_a(self.s,scope='eval',trainable=True)
			self.a_ = self.build_a(self.s_,scope='target',trainable=False)

		with tf.variable_scope('Critic'):
			q = self.build_c(self.s,self.a,scope='eval',trainable=True)
			q_ = self.build_c(self.s_,self.a_,scope='target',trainable=False)

		self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/target')
		self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/eval')
		self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Critic/target')
		self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Critic/eval')
		self.soft_replace = [tf.assign(t,(1-self.tau) * t + self.tau *e ) 
											for t,e in zip(self.at_params + self.ct_params ,self.ae_params + self.ce_params)]

		with tf.variable_scope('target_q'):
			target_q = self.r + self.gamma * q_ 

		with tf.variable_scope('td_error'):
			td_error = tf.reduce_mean(tf.squared_difference(target_q,q))
			#td_error = tf.losses.mean_squared_error(labels=target_q, predictions=q)

		with tf.variable_scope('c_train'):
			self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error,var_list=self.ce_params)

		with tf.variable_scope('a_train'):
			a_loss = -tf.reduce_mean(q) # maximize the q
			self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss,var_list=self.ae_params)

		self.sess.run(tf.global_variables_initializer())




	def build_a(self,s,scope,trainable):
		with tf.variable_scope(scope):
			# network weights
			w_initializer = tf.random_normal_initializer(0.,0.3)
			b_initializer = tf.constant_initializer(0.1)

			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1',[self.s_dim,30],initializer=w_initializer,trainable=trainable)
				b1 = tf.get_variable('b1',[30],initializer=b_initializer,trainable=trainable)
				l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

			with tf.variable_scope('action'):
				w2 = tf.get_variable('w2',[30,self.a_dim],initializer=w_initializer,trainable=trainable)
				b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,trainable=trainable)
				a = tf.nn.tanh(tf.matmul(l1,w2)+b2)

		return tf.multiply(a,self.a_bound,name='scaled_a') # Scale output to -action_bound to action_bound


	def build_c(self,s,a,scope,trainable):
		with tf.variable_scope(scope):
			# network weights
			w_initializer = tf.random_normal_initializer(0.,0.1)
			b_initializer = tf.constant_initializer(0.1)

			with tf.variable_scope('l1'):
				w1_s = tf.get_variable('w1_s_c',[self.s_dim,30],initializer=w_initializer,trainable=trainable)
				w1_a = tf.get_variable('w1_s_a',[self.a_dim,30],initializer=w_initializer,trainable=trainable)
				b1 = tf.get_variable('b1_c',[30],initializer=b_initializer,trainable=trainable)
				l1 = tf.nn.relu(tf.matmul(s,w1_s) + tf.matmul(a,w1_a) + b1)

			with tf.variable_scope('q'):
				w2 = tf.get_variable('w2_c',[30,1],initializer=w_initializer,trainable=trainable)
				b2 = tf.get_variable('b2_c',[1],initializer=b_initializer,trainable=trainable)
				q  = tf.matmul(l1,w2) + b2
		return q

	def train_target(self):
		#at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/target/params')
		self.sess.run(self.soft_replace)
		
	
	def store_transition(self,s,a,r,s_,done):
		transition = (s,a,[r],s_,done)
		#print('transition:',transition)
		if self.memory_count < self.memory_size:
			self.memory.append(transition)
			self.memory_count += 1
		else:
			self.memory.popleft()
			self.memory.append(transition)

	def choose_action(self,s):
		return self.sess.run(self.a,feed_dict={self.s:s.reshape(-1,self.s_dim)})[0]

	def learn(self):
		# soft target replacement
		self.train_target()

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

		#print(rewards)

		self.sess.run(self.atrain,feed_dict={self.s:states})
		self.sess.run(self.ctrain,feed_dict={self.s:states,self.a:actions,self.r:rewards,self.s_:next_states})
		# print('\n')
		# print(tf.trainable_variables())

