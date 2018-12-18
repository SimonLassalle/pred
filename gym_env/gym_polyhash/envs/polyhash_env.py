import os, subprocess, time, signal
import gym
from gym import error, spaces
from random import randint

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

        self._seed = randint(0, 200)
        self.observation_space = spaces.Box(low=0,
                                            high=self.number_of_actions,
                                            shape=(self.window_width, self.window_height))
        # TODO: add window movement in action_space
        self.action_space = spaces.Tuple([spaces.Discrete(self.number_of_actions),
                                            spaces.Discrete(self.window_width),
                                            spaces.Discrete(self.window_height)])

    def step(self, action):
        self._take_action(action)
        reward = self._get_reward()
        ob = self.getObservationSpace()
        episode_over = False
        return ob, reward, episode_over, {}

    def getObservationSpace(self):
        """
        Transform the """
        new_observation = []#self.env.cellsId[:]
        for x in range(len(self.env.cellsId)):
            new_observation.append([])
            for y in range(len(self.env.cellsId[0])):
                if self.env.cellsId[x][y] != -1:
                    new_observation.append(self.env.buildings[self.env.cellsId[x][y]].project.id)
        return new_observation


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
        self.reward = 0
        self.env.reset()
        return self.getObservationSpace()

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        print(self.env.cellsId)
