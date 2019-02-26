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

# Create the gym environment
ENV_NAME = 'Polyhash-v0'
env = gym.make(ENV_NAME)

# Parameters for the neural network
input_shape = env.observation_space.shape
nb_neuron_input = sum(input_shape)

output_shape = env.action_space.shape
nb_neuron_output = sum(output_shape)

# Create the model
model = Sequential()
model.add(Dense(nb_neuron_input, input_shape = (1,) + input_shape))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dense(nb_neuron_output, activation='softmax'))
model.summary()

# Create the DQN agent with the memory and policy hyper-parameters
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_neuron_output, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae', 'acc'])

# Create the metrics and pass it to the model as a callback for training
metrics = Metrics(dqn, env)
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2, callbacks=[metrics])

# the filename has the form modelName_memoryType|memorySize_policy_numberOfSteps
fileName = '1D__Sequential50000_BoltzmannQ_1000000steps(0)'

# Write the metrics into a file for post treatment
f1=open('./output/' + fileName + '.txt', 'w+')
f1.write(metrics.export_to_text())
f1.close()

# Optionally export figures
metrics.export_figs(fileName)

# Save network weights for later tests
dqn.save_weights('./output/' + fileName + '.h5f', overwrite=True)

# Test once after the training
dqn.test(env, nb_episodes=1, visualize=True)
