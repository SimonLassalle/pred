import numpy as np
import gym
import random
import os

import tensorflow as tf

from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input, Concatenate, Lambda
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory

from functools import reduce
from Metrics import Metrics

import gym_env.gym_polyhash.envs.polyhash_env

ENV_NAME = 'Polyhash-v0'

env = gym.make(ENV_NAME)

map_size = env.window_width * env.window_height

input_shape = env.observation_space.shape
nb_neuron_input = sum(input_shape)

output_shape = env.action_space.shape
nb_neuron_output = sum(output_shape)

inputs = Input(shape = (1,) + input_shape)
inputs_flatten = Flatten()(inputs)

def map_split(x, map_size = map_size):
    return x[:, 0: map_size]
def position_split(x, map_size = map_size, nb_neuron_input = nb_neuron_input):
    return x[:, map_size:nb_neuron_input]

map = Lambda(map_split)(inputs_flatten)
position = Lambda(position_split)(inputs_flatten)

map = Dense(map_size * 2)(map)
map = Activation('tanh')(map)
map = Dense(map_size)(map)
map = Activation('tanh')(map)

merged = Concatenate()([map, position])
merged = Dense(nb_neuron_input * 2, activation = 'tanh')(merged)
merged = Dense(nb_neuron_input, activation = 'tanh')(merged)
merged = Dense(nb_neuron_output, activation='softmax')(merged)

model = Model(inputs=[inputs],outputs=[merged])
model.summary()
model.compile(Adam(), loss='mean_squared_error')

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=4, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae', 'accuracy'])

for i in range(0, 8):
    metrics = Metrics(dqn, env)
    dqn.fit(env, nb_steps=10000, visualize=False, verbose=2, callbacks=[metrics])

    fileName = '1D_advanced_Sequential50000_BoltzmannQPolicy_10000steps(' + str(i) + ')'

    f1=open('./output/' + fileName + '.txt', 'w+')
    f1.write(metrics.export_to_text())
    f1.close()

    metrics.export_figs(fileName)

    dqn.save_weights('./output/' + fileName + '.h5f', overwrite=True)

    dqn.test(env, nb_episodes=1, visualize=False)
