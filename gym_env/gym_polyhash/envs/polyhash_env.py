import os, subprocess, time, signal
import gym
from gym import error, spaces

import referee as hash

class PolyhashEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = hash.Plan('data_small/a_example.in')
        self.number_of_actions = self.env.number_of_building_projects
        self.window_width = self.env.nb_rows
        self.window_height = self.env.nb_columns
        self.bad_action = False
        self.previous_score = 0

        self.reward = 0
        self.action = None

        self.number_of_steps = 0
        self.max_number_of_steps = 100

        self._seed = 123
        self.observation_space = spaces.Box(low=0,
                                            high=self.number_of_actions,
                                            shape=(self.window_width, self.window_height))
        self.action_space = spaces.Tuple([spaces.Discrete(self.number_of_actions),
                                            spaces.Discrete(self.window_width),
                                            spaces.Discrete(self.window_height)])

    def step(self, action):
        self._take_action(action)
        reward = self._get_reward()
        ob = self.env.cellsId
        episode_over = self.number_of_steps >= self.max_number_of_steps
        self.number_of_steps += 1
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        id = action[0]
        column = action[1]
        row = action[2]
        if self.env.canPlaceBuilding(id, row, column):
            self.bad_action = False
            building = self.env.createBuilding(id, row, column)
            self.env.placeBuilding(building)
        else:
            self.bad_action = True

        """ Converts the action space into an HFO action.
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == hfo_py.TURN:
            self.env.act(action_type, action[3])
        elif action_type == hfo_py.KICK:
            self.env.act(action_type, action[4], action[5])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(hfo_py.NOOP)"""

    def _get_reward(self):
        """ Reward is the difference of scores between two steps,
        or -10 if a bad action has been chosen. """
        if self.bad_action == True:
            return -10
        self.env.calcScore()
        reward = self.env.score - self.previous_score
        self.previous_score = self.env.score
        self.reward = reward
        return reward

    def reset(self):
        """ Set the environment's cells arrays to initial values. """
        self.previous_score = 0
        self.bad_action = False
        self.number_of_steps = 0
        self.reward = 0
        self.env.reset()

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        print(self.env.cellsVal)
