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
from Metrics import Metrics

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
model.add(Dense(nb_neuron_output, activation='tanh'))
model.summary()

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=4, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

metrics = Metrics(dqn)
dqn.fit(env, nb_steps=100, visualize=True, verbose=2, callbacks=[metrics])

print(metrics)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

#dqn.test(env, nb_episodes=1, visualize=True)
