#!/usr/bin/env python
#-*- coding: utf-8 -*-

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ddqn_truth import DDQN 


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):

    natural_DQN = DDQN(s_dim = env.observation_space.shape[0],
             a_dim = ACTION_SPACE,
             learning_rate = 0.005,
             e_greedy = 0.9,
             replace_target_iter = 200,
             memory_size = MEMORY_SIZE,
             e_greedy_increment = 0.001,
             double_q = False,
             sess = sess)

with tf.variable_scope('Double_DQN'):

    double_DQN = DDQN(s_dim = env.observation_space.shape[0],
             a_dim = ACTION_SPACE,
             learning_rate = 0.005,
             e_greedy = 0.9,
             replace_target_iter = 200,
             memory_size = MEMORY_SIZE,
             e_greedy_increment = 0.001,
             double_q = True,
             sess = sess)


sess.run(tf.global_variables_initializer())

def train(RL):
    total_steps = 0
    s = env.reset()

    while True:
        env.render()
        a = RL.choose_action(s)
        f_action = (a-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
        s_, r, done, info = env.step(np.array([f_action]))

        r /= 10 # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.
        RL.store_transition(s,a,r,s_,done)

        if total_steps > MEMORY_SIZE: #learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000: # stop game
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




