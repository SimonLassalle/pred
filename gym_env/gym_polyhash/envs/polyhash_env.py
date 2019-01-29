import os, subprocess, time, signal
import gym
from gym import error, spaces
from random import randint
import numpy as np

import referee as hash

class PolyhashEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = hash.Plan('data_small/a_example.in')
        self.number_of_building_projects = self.env.number_of_building_projects
        self.window_width = self.env.nb_rows
        self.window_height = self.env.nb_columns
        self.bad_action = False
        self.previous_score = 0

        self.reward = 0
        self.action = None

        self._seed = randint(0, 200)

        self.position_building_placement = np.array((0,0))
        self.observation_space = self.get_observation_space_1D()
        self.action_space = spaces.Discrete(self.number_of_building_projects + 1)

        print("ENV VARIABLES :")
        print("     number_of_building_projects :", self.number_of_building_projects)
        print("     window_width :", self.window_width)
        print("     window_height :", self.window_height)
        #print("     observation_space sample :", self.observation_space.sample())
        print("     observation_space shape :", self.observation_space.shape)
        #print("     action_space sample :", self.action_space.sample())
        print("     action_space shape :", self.action_space.shape)


    def get_observation_space_2D(self):
        observation_space = []
        for x in range(self.window_height) :
            observation_space_line = []
            for y in range(self.window_width) :
                observation_space_line.append(spaces.Discrete(self.number_of_building_projects))
            observation_space.append(spaces.Tuple(observation_space_line))

        tuple_position = spaces.Tuple(spaces.Discrete(self.window_width),
                                      spaces.Discrete(self.window_height))
        return spaces.Tuple((spaces.Tuple(observation_space), tuple_position))

    def get_observation_space_1D(self):
        observation_space = []
        for x in range(self.window_height) :
            for y in range(self.window_width) :
                observation_space.append(spaces.Discrete(self.number_of_building_projects))

        # Adding tuple representing position_building_placement
        observation_space.append(spaces.Discrete(self.window_width))
        observation_space.append(spaces.Discrete(self.window_height))
        return spaces.Tuple(observation_space)

    def step(self, action):
        print()
        print('POSITION :', self.position_building_placement)
        print("ACTION AT STEP :", action)
        self._take_action(action)
        reward = self._get_reward(action)
        print("REWARD :", reward)
        ob = self.getObservationSpace1D()
        self._update_position_building_placement()
        episode_over = self.position_building_placement[1] > self.window_height
        return ob, reward, episode_over, {}

    def getObservationSpace2D(self):
        new_observation = []#self.env.cellsId[:]
        for x in range(len(self.env.cellsId)):
            new_observation.append([])
            for y in range(len(self.env.cellsId[0])):
                if self.env.cellsId[x][y] != -1:
                    new_observation[x].append(self.env.buildings[self.env.cellsId[x][y]].project.id)
                else:
                    new_observation[x].append(-1)
        return new_observation, self.position_building_placement

    def getObservationSpace1D(self):
        new_observation = []#self.env.cellsId[:]
        for x in range(self.window_width):
            for y in range(self.window_height):
                if self.env.cellsId[x][y] != -1:
                    new_observation.append(self.env.buildings[self.env.cellsId[x][y]].project.id)
                else:
                    new_observation.append(-1)
        return new_observation + list(self.position_building_placement)


    def _take_action(self, action):
        if action != 3:
            id = action
            row = self.position_building_placement[0]
            column = self.position_building_placement[1]
            if self.env.canPlaceBuilding(id, row, column):
                self.bad_action = False
                building = self.env.createBuilding(id, row, column)
                self.env.placeBuilding(building)
            else:
                self.bad_action = True
        # else : the agent does nothing

    def _get_reward(self, action):
        """ Reward is the difference of scores between two steps,
        or -10 if a bad action has been chosen. """
        if self.bad_action == True:
            return -2
        if action == self.number_of_building_projects:
            return -1
        self.env.calcScore()
        reward = self.env.score - self.previous_score
        print("previous_score :", self.previous_score)
        print("actual_score :", self.env.score)
        print("reward :", reward)
        self.previous_score = self.env.score
        self.reward = reward
        return reward

    def _update_position_building_placement(self):
        self.position_building_placement = tuple(map(lambda x,y : x + y,self.position_building_placement,(1,0)))
        if self.position_building_placement[0] > self.window_width:
            self.position_building_placement = (0, self.position_building_placement[1]+1)
        return

    def reset(self):
        """ Set the environment's cells arrays to initial values. """
        self.previous_score = 0
        self.bad_action = False
        self.reward = 0
        self.position_building_placement = (0,0)
        self.env.reset()
        return np.concatenate([np.full(self.window_width * self.window_height, -1),self.position_building_placement])

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        print(self.env.cellsId)
