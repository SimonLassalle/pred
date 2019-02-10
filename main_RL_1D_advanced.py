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
from Metrics_Test import Metrics

import gym_env.gym_polyhash.envs.polyhash_env

# Create the gym environment
ENV_NAME = 'Polyhash-v0'
env = gym.make(ENV_NAME)

# Parameters for the neural network
map_size = env.window_width * env.window_height

input_shape = env.observation_space.shape
nb_neuron_input = sum(input_shape)

output_shape = env.action_space.shape
nb_neuron_output = sum(output_shape)

inputs = Input(shape = (1,) + input_shape)
inputs_flatten = Flatten()(inputs)

# Split the observation into tow separated inputs: the map and the posision
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

# Do the oposite: concatenate the outputs into one action
merged = Concatenate()([map, position])
merged = Dense(nb_neuron_input * 2, activation = 'tanh')(merged)
merged = Dense(nb_neuron_input, activation = 'tanh')(merged)
merged = Dense(nb_neuron_output, activation='softmax')(merged)

# Create the model
model = Model(inputs=[inputs],outputs=[merged])
model.summary()
model.compile(Adam(), loss='mean_squared_error')

# Create the DQN agent with the memory and policy hyper-parameters
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_neuron_output, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae', 'accuracy'])

# We execute 8 times the training to get workable metrics
for i in range(0, 8):
    # Create the metrics and pass it to the model as a callback for training
    metrics = Metrics(dqn, env)
    dqn.fit(env, nb_steps=10000, visualize=False, verbose=2, callbacks=[metrics])

    # the filename has the form modelName_memoryType|memorySize_policy_numberOfSteps
    fileName = '1D_advanced_Sequential50000_BoltzmannQPolicy_10000steps(' + str(i) + ')'

    # Write the metrics into a file for post treatment
    f1=open('./output/' + fileName + '.txt', 'w+')
    f1.write(metrics.export_to_text())
    f1.close()

    # Optionally export figures
    metrics.export_figs(fileName)

    # Save network weights for later tests
    dqn.save_weights('./output/' + fileName + '.h5f', overwrite=True)

    # Test once after the training
    dqn.test(env, nb_episodes=1, visualize=False)
