#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf 

np.random.seed(2)
tf.set_random_seed(2)

class Actor(object):
	def __init__(self,s_dim,a_dim,sess,learning_rate=0.001):
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.lr = learning_rate

		self.sess = sess

		self.build_softmax_network()


	def build_softmax_network(self):
		# input layer
		self.s = tf.placeholder(tf.float32,[1,self.s_dim],name='state') #input
		self.a = tf.placeholder(tf.int32,None,name="act")
		self.td_error = tf.placeholder(tf.float32,None,name="td_error")

		# network weights
		w_initializer = tf.random_normal_initializer(0.,0.1)
		b_initializer = tf.constant_initializer(0.1)

		c_names = ['actor_network',tf.GraphKeys.GLOBAL_VARIABLES]

		# fc1  hidden layers
		with tf.variable_scope('fc1'):
			w1 = tf.get_variable('w1',[self.s_dim,20],initializer=w_initializer,collections=c_names)
			b1 = tf.get_variable('b1',[20],initializer=b_initializer,collections=c_names)
			fc1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

		# fc2 softmax layer
		with tf.variable_scope('all_act'):
			w2 = tf.get_variable('w2',[20,self.a_dim],initializer=w_initializer,collections=c_names)
			b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,collections=c_names)
			all_act  = tf.matmul(fc1,w2) + b2

		# softmax output
		self.all_act_prob = tf.nn.softmax(all_act,name='act_prob') 

		with tf.variable_scope('exp_v'):
			#neg_log_prob = tf.log(self.all_act_prob[0, self.a])
			neg_log_prob = tf.reduce_sum(tf.log(self.all_act_prob)*tf.one_hot(self.a, self.a_dim))
			self.exp_v = tf.reduce_mean(neg_log_prob * self.td_error)

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)

	def choose_action(self,state):
		probs = self.sess.run(self.all_act_prob,feed_dict={self.s:state.reshape(-1,self.s_dim)})
		action = np.random.choice(range(probs.shape[1]),p=probs.ravel())# select action w.r.t the actions prob
		return action

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

				# fc1  hidden layers
		with tf.variable_scope('critic'):
			w1 = tf.get_variable('w1_c',[self.s_dim,20],initializer=w_initializer,collections=c_names)
			b1 = tf.get_variable('b1_c',[20],initializer=b_initializer,collections=c_names)
			fc1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

		# fc2 softmax layer
		with tf.variable_scope('V'):
			w2 = tf.get_variable('w2_c',[20,1],initializer=w_initializer,collections=c_names)
			b2 = tf.get_variable('b2_c',[1],initializer=b_initializer,collections=c_names)
			self.v  = tf.matmul(fc1,w2) + b2


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
