import numpy as np
import gym
import random

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from functools import reduce
from Metrics_Test import Metrics

import gym_env.gym_polyhash.envs.polyhash_env

ENV_NAME = 'Polyhash-v0'

env = gym.make(ENV_NAME)

input_shape = env.observation_space.shape
nb_neuron_input = sum(input_shape)

output_shape = env.action_space.shape
nb_neuron_output = sum(output_shape)

model = Sequential()
model.add(Dense(nb_neuron_input, input_shape = (1,) + input_shape))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dense(nb_neuron_output, activation='softmax'))
model.summary()

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=4, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae', 'acc'])

metrics = Metrics(dqn, env)
#fileName = '1D_advanced_Sequential1000_BoltzmannQ_10000steps(7)'
#fileName = '1D_advanced_Sequential1000_EpsGreedyQ_10000steps(7)'
#fileName = '1D_advanced_Sequential1000_MaxBoltzmannQ_10000steps(7)'
#fileName = '1D_advanced_Sequential50000_BoltzmannQPolicy_10000steps(7)'
#fileName = '1D_advanced_Sequential50000_MaxBoltzmannQ_1000000steps(0)'
fileName = '1D__Sequential50000_BoltzmannQ_1000000steps(0)'

dqn.load_weights('./output/' + fileName + '.h5f')
dqn.test(env, nb_episodes=1, visualize=False, callbacks=[metrics])

metrics.export_figs(fileName)

cumulated_reward = metrics.cumulated_reward()
import matplotlib.pyplot as plt
plt.figure()
plt.plot(cumulated_reward, alpha = .6)
plt.title('cumulated_reward for ' + fileName)
plt.ylabel('cumulated_reward')
plt.xlabel('steps')
plt.savefig('./metrics/' + fileName + '_cumulated_reward.png')
