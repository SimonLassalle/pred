import numpy as np
import gym

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
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

print("model init")
model = Sequential()
print("adding Flatten-1")
model.add(Flatten(input_shape=(1,4,7)))
print("adding Dense-1")
model.add(Dense(64))
print("adding Dense-2")
model.add(Dense(4))

model.summary()

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=4, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=100, visualize=True, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

dqn.test(env, nb_episodes=1, visualize=True)
