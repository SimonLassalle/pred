import os, subprocess, time, signal
import gym
from gym import error, spaces

from env_source import Plan

import logging
logger = logging.getLogger(__name__)

class PolyhashEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = None
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=1)
        # Action space omits the Tackle/Catch actions, which are useful on defense
        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1)))

    def _step(self, action):
        self._take_action(action)
        reward = self._get_reward()
        ob = None
        episode_over = False
        return ob, reward, episode_over, {}

    def _take_action(self, action):
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
        pass

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        pass

    def _reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        pass

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        print("coucou")
