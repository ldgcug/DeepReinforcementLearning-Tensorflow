#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf 

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient(object):

	def __init__(self,
				 s_dim,
				 a_dim,
				 learning_rate = 0.01,
				 reward_decay = 0.95,
				 output_graph = False
	):
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.lr = learning_rate
		self.gamma = reward_decay

		self.ep_obs,self.ep_as,self.ep_rs = [],[],[]

		self.build_net()

		self.sess = tf.Session()

		if output_graph:
			tf.summary.FileWriter("logs/",self.sess.graph)

		self.sess.run(tf.global_variables_initializer())

	def build_net(self):

		# input layer
		self.s = tf.placeholder(tf.float32,[None,self.s_dim],name='s_dim') #input
		self.tf_acts = tf.placeholder(tf.int32,[None,],name="actions_num")
		self.tf_vt = tf.placeholder(tf.float32,[None,],name="actions_value")

		# network weights
		w_initializer = tf.random_normal_initializer(0.,0.3)
		b_initializer = tf.constant_initializer(0.1)

		c_names = ['softmax_output',tf.GraphKeys.GLOBAL_VARIABLES]

		# fc1  hidden layers
		with tf.variable_scope('fc1'):
			w1 = tf.get_variable('w1',[self.s_dim,20],initializer=w_initializer,collections=c_names)
			b1 = tf.get_variable('b1',[20],initializer=b_initializer,collections=c_names)
			fc1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

		# fc2 softmax layer
		with tf.variable_scope('fc2'):
			w2 = tf.get_variable('w2',[20,self.a_dim],initializer=w_initializer,collections=c_names)
			b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,collections=c_names)
			all_act  = tf.matmul(fc1,w2) + b2

		# softmax output
		self.all_act_prob = tf.nn.softmax(all_act,name='act_prob')  # use softmax to convert to probability

		with tf.variable_scope('loss'):
			# to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,labels=self.tf_acts)
			# or in this way
			# neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
			loss = tf.reduce_mean(neg_log_prob * self.tf_vt) # reward guided loss

		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def choose_action(self,state):
		prob_weights = self.sess.run(self.all_act_prob,feed_dict={self.s:state.reshape(-1,self.s_dim)})
		action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())# select action w.r.t the actions prob
		return action

	def store_transition(self,s,a,r):
		self.ep_obs.append(s)
		self.ep_as.append(a)
		self.ep_rs.append(r)

	def learn(self):
		# discount and normalize episode reward
		discounted_ep_rs_norm = self.discount_and_norm_rewards()

		# train on episode
		self.sess.run(self.train_op,feed_dict={
					self.s:np.vstack(self.ep_obs), # shape=[None, n_obs]
					self.tf_acts:np.array(self.ep_as),  # shape=[None, ]
					self.tf_vt:discounted_ep_rs_norm  # shape=[None, ]
			})
		self.ep_obs,self.ep_as,self.ep_rs = [],[],[] # empty episode data
		return discounted_ep_rs_norm

	def discount_and_norm_rewards(self):
		# discount episode rewards
		discounted_ep_rs = np.zeros_like(self.ep_rs)
		running_add = 0
		for t in reversed(range(0,len(self.ep_rs))):
			running_add = running_add * self.gamma + self.ep_rs[t]
			discounted_ep_rs[t] = running_add

		# normalize episode rewards
		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs










