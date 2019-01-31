import numpy as np
import gym

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from functools import reduce

import gym_env.gym_polyhash.envs.polyhash_env

ENV_NAME = 'Polyhash-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

model_map = Sequential()
model_map.add(Flatten(input_shape=(1,4,7)))
model_map.add(Dense(64, activation='tanh'))
model_map.add(Activation('tanh'))

model_position = Sequential()
model_position.add(Dense(2,input_shape=(2,)))

merged = Concatenate()([model_map.output, model_position.output])
out = Dense(66, activation='tanh')(merged)
out = Dense(4, activation='softmax')(out)

model = Model([model_map.input, model_position.input], [out])
model.summary()

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=4, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=100, visualize=True, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

dqn.test(env, nb_episodes=1, visualize=True)
