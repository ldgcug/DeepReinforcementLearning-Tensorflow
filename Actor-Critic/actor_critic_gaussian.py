#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf 

np.random.seed(2)
tf.set_random_seed(2)

class Actor(object):
	def __init__(self,s_dim,action_bound,sess,learning_rate=0.001):
		self.s_dim = s_dim
		self.action_bound = action_bound
		self.lr = learning_rate

		self.sess = sess

		self.build_gaussian_network()

	def build_gaussian_network(self):
		# input layer
		self.s = tf.placeholder(tf.float32,[1,self.s_dim],name='state') #input
		self.a = tf.placeholder(tf.float32,None,name="act")
		self.td_error = tf.placeholder(tf.float32,None,name="td_error")

		# network weights
		w_initializer = tf.random_normal_initializer(0.,0.1)
		b_initializer = tf.constant_initializer(0.1)

		c_names = ['actor_network',tf.GraphKeys.GLOBAL_VARIABLES]

		# fc1  hidden layers
		with tf.variable_scope('l1'):
			w1 = tf.get_variable('w1',[self.s_dim,30],initializer=w_initializer,collections=c_names)
			b1 = tf.get_variable('b1',[30],initializer=b_initializer,collections=c_names)
			l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

		#mu
		with tf.variable_scope('mu'):
			w2 = tf.get_variable('w2',[30,1],initializer=w_initializer,collections=c_names)
			b2 = tf.get_variable('b2',[1],initializer=b_initializer,collections=c_names)
			mu = tf.nn.tanh(tf.matmul(l1,w2)+b2)

		#sigma
		with tf.variable_scope('sigma'):
			w3 = tf.get_variable('w3',[30,1],initializer=w_initializer,collections=c_names)
			b3 = tf.get_variable('b3',[1],initializer=b_initializer,collections=c_names)
			sigma = tf.nn.softplus(tf.matmul(l1,w3)+b3)

		global_step = tf.Variable(0,trainable=False)
		# self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
		self.mu,self.sigma = tf.squeeze(mu*2),tf.squeeze(sigma+0.1)
		self.normal_dist = tf.distributions.Normal(self.mu,self.sigma)

		self.action = tf.clip_by_value(self.normal_dist.sample(1),self.action_bound[0],self.action_bound[1])

		with tf.variable_scope('exp_v'):
			log_prob = self.normal_dist.log_prob(self.a)
			self.exp_v = log_prob * self.td_error
			self.exp_v += 0.01 * self.normal_dist.entropy()

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v,global_step)

	def choose_action(self,state):
		return self.sess.run(self.action,feed_dict={self.s:state.reshape(-1,self.s_dim)})

	def learn(self,s,a,td):
		s = s.reshape(-1,self.s_dim)
		_ = self.sess.run(self.train_op,feed_dict={self.s:s,self.a:a,self.td_error:td})

class Critic(object):
	def __init__(self,s_dim,sess,learning_rate=0.01,reward_decay=0.9):
		self.s_dim = s_dim
		self.lr = learning_rate
		self.gamma = reward_decay

		self.sess = sess 

		self.build_critic_network()

	def build_critic_network(self):

		self.s = tf.placeholder(tf.float32,[1,self.s_dim],"state")
		self.v_ = tf.placeholder(tf.float32,[1,1],"v_next")
		self.r = tf.placeholder(tf.float32,None,"r")

		w_initializer = tf.random_normal_initializer(0.,0.1)
		b_initializer = tf.constant_initializer(0.1)

		c_names = ['critic_network',tf.GraphKeys.GLOBAL_VARIABLES]

		with tf.variable_scope('critic'):
			w1 = tf.get_variable('w1_c',[self.s_dim,30],initializer=w_initializer,collections=c_names)
			b1 = tf.get_variable('b1_c',[30],initializer=b_initializer,collections=c_names)
			l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

		with tf.variable_scope('V'):
			w2 = tf.get_variable('w2_c',[30,1],initializer=w_initializer,collections=c_names)
			b2 = tf.get_variable('b2_c',[1],initializer=b_initializer,collections=c_names)
			self.v  = tf.matmul(l1,w2) + b2

		with tf.variable_scope('squard_TD_error'):
			self.td_error = self.r + self.gamma * self.v_ - self.v
			self.loss = tf.square(self.td_error)

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


	def learn(self,s,r,s_):
		s = s.reshape(-1,self.s_dim)
		s_ = s_.reshape(-1,self.s_dim)
		v_ = self.sess.run(self.v,feed_dict={self.s:s_})

		td_error = self.sess.run(self.td_error,feed_dict={self.s:s,self.v_:v_,self.r:r})
		_ = self.sess.run(self.train_op,feed_dict={self.s:s,self.v_:v_,self.r:r})

		return td_error

