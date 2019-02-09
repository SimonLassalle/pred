import os, subprocess, time, signal
import gym
from gym import error, spaces
from random import randint
import numpy as np

import referee as hash

class PolyhashEnv(gym.Env):
    """ This class describe the environement of PolyHash 2018. It is created to \
    be used with Keras-RL and a neural network.\
    There is two implementation : one in 1D and another one in 2D.\
    There is no automatic procedure to switch between these two modes. It needs \
    to be manually done. """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """ Initialisation function. """
        # The env variable is an instantiation of the refree developed by
        # Mr. PERREIRA DA SILVA
        self.env = hash.Plan('data_small/a_example.in')
        # number_of_building_projects describe the number of different building
        # there is.
        self.number_of_building_projects = self.env.number_of_building_projects
        # window_width and window_height represent the size of the convolution.
        self.window_width = (self.env.nb_rows if self.env.nb_rows < 100 else 100)
        self.window_height = (self.env.nb_columns if self.env.nb_columns < 100 else 100)
        self.bad_action = False
        self.previous_score = 0

        self.reward = 0
        self.action = None

        # Seed for random usage.
        self._seed = randint(0, 200)

        # Get parameters of the environment for interaction with the agent.
        self.position_building_placement = np.array((0,0))
        self.observation_space = self.get_observation_space_1D()
        self.action_space = self.get_action_space()


    def get_observation_space_2D(self):
        """ This function retuns the observation space for 2D networks. """
        observation_space = []
        for x in range(self.window_height) :
            observation_space_line = []
            for y in range(self.window_width) :
                observation_space_line.append(spaces.Discrete(self.number_of_building_projects))
            observation_space.append(spaces.Tuple(observation_space_line))

        # Adding tuple representing position_building_placement
        tuple_position = spaces.Tuple(spaces.Discrete(self.window_width),
                                      spaces.Discrete(self.window_height))
        return spaces.Tuple((spaces.Tuple(observation_space), tuple_position))

    def get_observation_space_1D(self):
        """ This function retuns the observation space for 1D networks. """
        observation_space = []
        for x in range(self.window_height) :
            for y in range(self.window_width) :
                observation_space.append(spaces.Discrete(self.number_of_building_projects))

        # Adding tuple representing position_building_placement
        observation_space.append(spaces.Discrete(self.window_width))
        observation_space.append(spaces.Discrete(self.window_height))

        observation_space = spaces.Tuple(observation_space)
        observation_space.shape = (self.window_height * self.window_width + 2,)
        return observation_space

    def get_action_space(self):
        """ This function retuns the action space. """
        action_space = spaces.Discrete(self.number_of_building_projects + 1)
        action_space.shape = (self.number_of_building_projects + 1,)
        return action_space

    def step(self, action):
        """ This function is called when the agent has decided which action to \
        perform. """
        self._take_action(action)
        reward = self._get_reward(action)
        ob = self.get_observation_1D()
        self._update_position_building_placement()
        episode_over = self.position_building_placement[1] > self.window_height - 1
        return ob, reward, episode_over, {}

    def get_observation_2D(self):
        """ This function retuns the observation for 2D networks. """
        new_observation = []
        for x in range(len(self.env.cellsId)):
            new_observation.append([])
            for y in range(len(self.env.cellsId[0])):
                if self.env.cellsId[x][y] != -1:
                    new_observation[x].append(self.env.buildings[self.env.cellsId[x][y]].project.id)
                else:
                    new_observation[x].append(-1)

        # This part is to take only the convolution window
        # This is a naïve function
        # Since we used only the small map, we never had to test
        # Theses lines.
        if self.window_width == 100 and self.window_height == 100 :
            x = self.position_building_placement[0] - 50
            y = self.position_building_placement[1] - 50
            new_observation = new_observation[x:x+100,y:y+100]
        return new_observation, self.position_building_placement

    def get_observation_1D(self):
        """ This function retuns the observation for 1D networks. """
        observation = self.get_observation_2D()[0]
        new_observation = []
        for x in observation :
            new_observation += x
        return new_observation + list(self.position_building_placement)

    def _take_action(self, action):
        """ This function tries to do the action asked by the network. It \
        update the bad_action variable if the action is not feasible. """
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
        """ This function returns the reward. It is the difference of scores \
        between two steps or -1 if a bad action has been chosen. """
        if self.bad_action == True:
            self.reward = -1
            return self.reward
        if action == self.number_of_building_projects:
            self.reward = 0
            return self.reward
        self.env.calcScore()
        self.reward = self.env.score - self.previous_score
        self.previous_score = self.env.score
        return self.reward

    def _update_position_building_placement(self):
        """ This function move the convolution window. """
        self.position_building_placement = tuple(map(lambda x,y : x + y,self.position_building_placement,(1,0)))
        if self.position_building_placement[0] > self.window_width - 1:
            self.position_building_placement = (0, self.position_building_placement[1]+1)
        return

    def reset(self):
        """ Set the environment's cells arrays to initial values. """
        self.previous_score = 0
        self.bad_action = False
        self.reward = 0
        self.position_building_placement = np.array((0,0))
        self.action = None
        self._seed = randint(0, 200)
        self.env.reset()
        return np.concatenate([ np.full(self.window_width * self.window_height, -1),
                                self.position_building_placement])

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        print(self.env.cellsId)
