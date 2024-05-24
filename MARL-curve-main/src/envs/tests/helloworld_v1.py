import copy

import cv2
import gym
from PIL import Image, ImageDraw
import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
import random
import torch
from utils.draw import draw_grid, fill_cell, write_cell_text
from smac.env.multiagentenv import MultiAgentEnv


class HelloWorld(MultiAgentEnv):
    """MultiAgentEnv / gym.Env, the base of customized envs"""

    def __init__(
            self,
            n_agents=4,
            grid_shape='8, 8',
            episode_limit: int = 10000,
            seed=None,
            render='human',
            step_reward=-0.01,
            collide_reward=-10
    ):
        # init
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        if seed:
            self._seed = seed
        self._render_mode = render

        # temp original grid steup
        self._original_grid = [
            "........", "........", "........", "...@@...", "...@@...", "........", "........", "........"
        ]
        self._agent_init_pos = {
            0: (7, 7), 1: (0, 0), 2: (7, 0), 3: (0, 7)
        }
        self._agent_goal_pos = {
            0: (0, 0), 1: (7, 7), 2: (0, 7), 3: (7, 0)
        }
        self._actions = [0, 1, 2, 3, 4]

        # init grid
        self._grid_shape = tuple(map(int, grid_shape.split(',')))
        self._n_agents = n_agents
        self.agents = [a for a in range(self._n_agents)]
        self.episode_limit = episode_limit
        self._step_count = None  # environment step counter
        self._full_obs = self.__create_grid()
        self._base_img = self.__draw_base_img()
        self.agent_pos = {a: None for a in self.agents}
        self._step_rew = step_reward
        self._collide_rew = collide_reward

    def reset(self):
        """ Returns initial observations and states """
        self._total_episode_reward = [0 for _ in range(self._n_agents)]
        self._step_count = 0
        self._agent_step_count = [0 for _ in range(self._n_agents)]
        self._agent_dones = [False for _ in range(self._n_agents)]
        self._curr_agents_count = 0
        self.agent_pos = deepcopy(self._agent_init_pos)
        self.__init_full_obs()
        return self.get_obs()

    def step(self, agents_action):
        """ Handle Transition. Returns reward, terminated, info """
        if isinstance(agents_action, torch.Tensor):
            agents_action = agents_action.detach().cpu().numpy()
        assert len(agents_action) == self._n_agents
        assert all([action_i in ACTION_MEANING.keys() for action_i in agents_action])
        self._step_count += 1  # global environment step
        rewards = [0 for _ in range(self._n_agents)]  # initialize rewards array
        step_collisions = 0  # counts collisions in this step
        # step for every agent
        for agent_i, action in enumerate(agents_action):
            if not self._agent_dones[agent_i]:
                self._agent_step_count[agent_i] += 1  # agent step count
                # checks if there is a collision; this is done in the __update_agent_pos method
                collision_flag = self.__update_agent_pos(agent_i, action)
                if collision_flag:
                    rewards[agent_i] += self._collide_rew
                    step_collisions += 1
                # gives additional step punishment to avoid jams
                rewards[agent_i] += self._step_rew
            self._total_episode_reward[agent_i] += rewards[agent_i]

            # checks if destination was reached
            if self.__reached_dest(agent_i):
                self._agent_dones[agent_i] = True
                self._curr_agents_count -= 1

            # if max_steps was reached, terminate the episode
            if self._step_count >= self.episode_limit:
                self._agent_dones[agent_i] = True

        # info stores the info about this transition
        info = {}
        info['_step_count'] = self._step_count
        return sum(rewards), self._agent_dones, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return np.array([self.get_obs_agent(agent_i) for agent_i in self.agents])

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return np.array(self._full_obs).flatten()  # flatten otherwise episode_buffer raise error

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self._grid_shape[0] * self._grid_shape[0]  # obs size for each agent

    def get_state(self):
        """ Returns the global state (all agent states) as a list """
        return np.array(self._full_obs).flatten()  # flatten otherwise episode_buffer raise error

    def get_state_size(self):
        """ Returns the shape of the global state """
        return self._grid_shape[0] * self._grid_shape[0]  # state size for the environment

    def get_avail_actions(self):
        """ Returns the available actions of all agents in a list """
        return [self.get_avail_agent_actions(i) for i in self.agents]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self._actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return len(self._actions)

    def render(self):
        """ Render something when have a monitor """
        img = copy.copy(self._base_img)
        for agent_i in range(self._n_agents):
            if not self._agent_dones[agent_i]:
                fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENTS_COLORS[agent_i])
                write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)
        img = np.asarray(img)
        if self._render_mode == 'rgb_array':
            return img
        elif self._render_mode == 'human':
            cv2.imshow('Gazebo_GridWorld', img)
            cv2.waitKey(1)
            return img

    def close(self):
        """ Close the environment """
        pass

    def seed(self):
        """ Set seed for the environment """
        pass

    def save_replay(self):
        """ Save a replay """
        pass

    def get_env_info(self):
        """ Return all key information about the environment """
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self._n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def episode_done(self):
        return sum(self._agent_dones) == self._n_agents

    def __is_valid(self, pos):
        # return if the pos is valid / inside the grid range
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def __is_cell_vacant(self, pos):
        # check if pos is available (safe) to move an agent into
        return self.__is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['ept'])

    def __is_cell_obstacle(self, pos):
        # check if pos is an obstacle
        return self.__is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['obs'])

    def __create_grid(self):
        # create a grid with every cell as wall
        _grid = [[PRE_IDS['ept'] for _ in range(self._grid_shape[1])] for _ in range(self._grid_shape[0])]
        for i in range(self._grid_shape[1]):
            for j in range(self._grid_shape[0]):
                _grid[i][j] = PRE_IDS['ept'] if self._original_grid[i][j] == '.' else PRE_IDS['obs']
        return _grid

    def __init_full_obs(self):
        # Initiates environment, put agents in the env
        self._full_obs = self.__create_grid()
        for agent_i in self.agents:
            self.__update_agent_view(agent_i)
        self.__draw_base_img()

    def __update_agent_view(self, agent_i):
        # put agent_i in the env
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = (agent_i + 1)

    def __draw_base_img(self):
        # create grid and make everything black
        img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill=WALL_COLOR)
        # draw tracks
        for i, row in enumerate(self._full_obs):
            for j, col in enumerate(row):
                if col == PRE_IDS['ept']:
                    fill_cell(img, (i, j), cell_size=CELL_SIZE, fill=(143, 141, 136), margin=0.05)
                elif col == PRE_IDS['obs']:
                    fill_cell(img, (i, j), cell_size=CELL_SIZE, fill=(242, 227, 167), margin=0.02)
        return img

    def __reached_dest(self, agent_i):
        # verifies if the agent_i reached a destination place, if reach, remove agent
        pos = self.agent_pos[agent_i]
        if pos == self._agent_goal_pos[agent_i]:
            self._full_obs[pos[0]][pos[1]] = PRE_IDS['ept']
            return True
        return False

    def __update_agent_pos(self, agent_i, action):
        # updates the agent position in the environment
        curr_pos = copy.copy(self.agent_pos[agent_i])
        if action == 0:  # LEFT
            next_pos = (curr_pos[0] - 1, curr_pos[1])
        elif action == 1:  # RIGHT
            next_pos = (curr_pos[0] + 1, curr_pos[1])
        elif action == 2:  # UP
            next_pos = (curr_pos[0], curr_pos[1] - 1)
        elif action == 3:  # DOWN
            next_pos = (curr_pos[0], curr_pos[1] + 1)
        elif action == 4:  # STAY
            return False
        else:
            raise Exception('Action Not found!')

        # if there is an agent out of range collision
        if not self.__is_valid(next_pos):
            return True

        # if there is an agent-obstacle collision
        if self.__is_cell_obstacle(next_pos):
            return True

        # if there is an agent-agent collision
        if not self.__is_valid(next_pos) or self.__check_collision(next_pos):
            return True

        # if there is no collision and the next position is free, updates agent position
        if self.__is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['ept']
            self.__update_agent_view(agent_i)
        return False

    def __check_collision(self, pos):
        # verifies if a transition to the position pos will result on a collision.
        # return self.__is_valid(pos) and self._full_obs[pos[0]][pos[1]].find(PRE_IDS['agent']) > -1
        return self.__is_valid(pos) and self._full_obs[pos[0]][pos[1]] > 0

CELL_SIZE = 30

WALL_COLOR = 'black'

AGENTS_COLORS = [
    "red",
    "blue",
    "yellow",
    "orange",
    "green"
]

ACTION_MEANING = {
    0: "LEFT",
    1: "RIGHT",
    2: "UP",
    3: "DOWN",
    4: "STAY"
}

PRE_IDS = {
    'obs': -1,
    'ept': 0,
    'agent': ''
}
