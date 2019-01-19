import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from functools import reduce

import gym_env.gym_polyhash.envs.polyhash_env

ENV_NAME = 'Polyhash-v0'


print()
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

input_nb_nodes = reduce((lambda x, y: x * y), env.observation_space.shape)
output_nb_nodes = 3#reduce((lambda x, y: x * y), env.action_space.shape[0])

print("observation_space.shape : ", env.observation_space.shape)
print("observation_space : ", env.observation_space)
print()
print("action_space.shape : ", env.action_space.shape)
print("action_space : ")
print()
print("actions_available : ", env.actions_available)
print()
print("input_nb_nodes : ", input_nb_nodes)
print("output_nb_nodes : ", output_nb_nodes)
print("(input_nb_nodes + output_nb_nodes)//2 : ", (input_nb_nodes + output_nb_nodes)//2)
print()

print("model init")
model = Sequential()
print("adding Dense-1")
model.add(Dense(7, input_shape=(1,) +env.observation_space.shape))
print("adding Activation-1")
model.add(Activation('relu'))
print("adding Flatten-1")
model.add(Flatten())
print("adding Dense-2")
model.add(Reshape((1, 28)))
print("adding Dense-3")
model.add(Dense(3))

model.summary()

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=output_nb_nodes, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=100, visualize=True, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

dqn.test(env, nb_episodes=5, visualize=True)
