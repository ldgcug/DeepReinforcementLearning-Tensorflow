#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math
import random
import numpy as np  
import tensorflow as tf 
from collections import deque  

np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):

	def __init__(self,capacity):
		self.capacity = capacity # for all priority-values
		self.tree = np.zeros(2 * capacity - 1) # Tree
		self.data = np.zeros(capacity,dtype=object) # for all transition

		self.size = 0
		self.curr_point = 0

	def add(self,data):
		self.data[self.curr_point] = data
		self.update(self.curr_point,max(self.tree[self.capacity-1:self.capacity+self.size]) + 1)

		self.curr_point += 1

		if self.curr_point >= self.capacity:
			self.curr_point = 0

		if self.size < self.capacity:
			self.size += 1


	def update(self,point,weight):
		tree_idx = point + self.capacity - 1
		change = weight - self.tree[tree_idx]

		self.tree[tree_idx] = weight

		parent = (tree_idx - 1) // 2
		while parent >= 0:
			self.tree[parent] += change
			parent = (parent -1) // 2

	def total_p(self):
		return self.tree[0]

	def get_min(self):
		return min(self.tree[self.capacity - 1: self.capacity + self.size -1])

	def sample(self,v):
		idx = 0
		while idx < self.capacity-1:
			l_idx = idx * 2 +1
			r_idx = l_idx +1
			if self.tree[l_idx] >= v:
				idx = l_idx 
			else:
				idx = r_idx 
				v = v - self.tree[l_idx]

		point = idx - (self.capacity - 1)

		return point,self.data[point],self.tree[idx] / self.total_p()


class Memory(object):

	def __init__(self,batch_size,max_size,beta):
		self.batch_size = batch_size
		#self.max_size = 2**math.floor(math.log2(max_size))
		self.beta = beta
		self.sum_tree = SumTree(max_size)

	def store(self,s,a,r,s_,done):
		transitions = (s,a,r,s_,done)
		self.sum_tree.add(transitions)

	def get_mini_batches(self):
		n_sample = self.batch_size if self.sum_tree.size >= self.batch_size else self.sum_tree.size 
		total = self.sum_tree.total_p()

		step = total // n_sample
		points_transitions_probs = []

		for i in range(n_sample):
			v = np.random.uniform(i * step,(i +1) * step -1)
			t = self.sum_tree.sample(v)
			points_transitions_probs.append(t)

		points,transitions,probs = zip(*points_transitions_probs)

		#max_importance_ratio = (n_sample * self.sum_tree.get_min())**-self.beta
		#print('max:',max_importance_ratio)
		mini_prob = self.sum_tree.get_min() / total
		importance_ratio = [pow(probs[i] /mini_prob,-self.beta) for i in range(len(probs))]
		#tuple(np.array(e) for e in zip(*transitions))
		return points,transitions,importance_ratio

	def update(self,points,td_error):
		for i in range(len(points)):
			td_error += 0.01
			self.sum_tree.update(points[i],td_error[i])

class DQNPrioritizedReplay:
	def __init__(self,
				 s_dim,
				 a_dim,
				 learning_rate = 0.005,
				 reward_decay = 0.9,
				 e_greedy = 0.9,
				 replace_target_iter = 500,
				 memory_size = 10000,
				 batch_size = 32,
				 e_greedy_increment = None,
				 output_graph = False,
				 prioritized = True,
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
		self.memory_count = 0
		self.memory_size = memory_size
		self.batch_size = batch_size

		self.prioritized = prioritized

		#toal learning step
		self.learn_step_counter = 0

		#consist of [target_net,evaluate_net]
		self.build_net()

		if self.prioritized:
			self.memory = Memory(batch_size,memory_size,0.9)
		else:
			self.memory = deque()

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

		if self.prioritized:
			self.importance_ratio = tf.placeholder(tf.float32,[None,1],name = 'importance_ratio')
		

		w_initializer = tf.random_normal_initializer(0.,0.3)
		b_initializer = tf.constant_initializer(0.1)

		with tf.variable_scope('eval_net'):
			c_names = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
			
			with tf.variable_scope('layer1'):
				w1 = tf.get_variable('w1',[self.s_dim,20],initializer=w_initializer,collections=c_names)
				b1 = tf.get_variable('b1',[20],initializer=b_initializer,collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

			with tf.variable_scope('layer2'):
				w2 = tf.get_variable('w2',[20,self.a_dim],initializer=w_initializer,collections=c_names)
				b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,collections=c_names)
				self.q_eval = tf.matmul(l1,w2) + b2

		with tf.variable_scope('loss'):
			if self.prioritized:
				self.td_error = tf.reduce_sum(tf.abs(self.q_target - self.q_eval),axis = 1)
				self.loss = tf.reduce_mean(self.importance_ratio * tf.squared_difference(self.q_target,self.q_eval))
			else:
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

			with tf.variable_scope('layer2'):
				w2 = tf.get_variable('w2',[20,self.a_dim],initializer=w_initializer,collections=c_names)
				b2 = tf.get_variable('b2',[self.a_dim],initializer=b_initializer,collections=c_names)
				self.q_next = tf.matmul(l1,w2) + b2

	def store_transition(self,s,a,r,s_,done):
		transition = (s,a,r,s_,done)
		if self.prioritized:
			self.memory.store(s,a,r,s_,done)
		else:
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

		if self.prioritized:
			points,mini_batch,importance_ratio = self.memory.get_mini_batches()
		else:
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
		# add q_eval_next
		q_target = q_eval.copy()


		for k in range(len(mini_batch)):
			if dones[k]:
				q_target[k][actions[k]] = rewards[k]
			else:
				# dqn
				q_target[k][actions[k]]	= rewards[k] + self.gamma * np.max(q_next[k])

		if self.prioritized:
			_ = self.sess.run(self.train_op,feed_dict={self.s:states,self.q_target:q_target,self.importance_ratio:np.array([importance_ratio]).T})
			td_error = self.sess.run(self.td_error,feed_dict={self.s:states,self.q_target:q_target,self.importance_ratio:np.array([importance_ratio]).T})
			loss = self.sess.run(self.loss,feed_dict={self.s:states,self.q_target:q_target,self.importance_ratio:np.array([importance_ratio]).T})
			self.memory.update(points,td_error)
		else:
			loss = self.sess.run(self.loss,feed_dict={self.s:states,self.q_target:q_target})
			_ = self.sess.run(self.train_op,feed_dict={self.s:states,self.q_target:q_target})
		
		self.cost_his.append(loss)

		# increasing epsilon
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1

	def train_target(self):
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.sess.run([tf.assign(t,e) for t,e in zip(t_params,e_params)])
