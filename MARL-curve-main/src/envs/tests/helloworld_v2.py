import copy
import os.path

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
from draw import draw_grid, fill_cell, write_cell_text
# from smac.env.multiagentenv import MultiAgentEnv


class HelloWorld(gym.Env):
    """MultiAgentEnv, the base of customized envs"""

    def __init__(
            self,
            grid_file_path='../mapf_baseline/mapf-map/random-64-64-20.map',
            agents_path='../mapf_baseline/scen-random/random-64-64-20-random-1.scen',
            n_agents=10,
            episode_limit: int = 10000,
            seed=None,
            render='human',
            step_reward=-0.01,
            collide_reward=-10
    ):
        # init
        assert os.path.exists(grid_file_path) and os.path.exists(agents_path)
        self._grid_file_path = grid_file_path
        self._agent_path = agents_path
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        if seed:
            self._seed = seed
        self._render_mode = render

        # init info
        self._n_agents = n_agents
        self.agents = [a for a in range(self._n_agents)]
        self._episode_limit = episode_limit
        self._step_count = None  # environment step counter

        # init grid
        self._grid_shape = None
        self._original_grid = None
        self._agent_init_pos = {a: None for a in self.agents}
        self._agent_goal_pos = {a: None for a in self.agents}
        self._actions = [0, 1, 2, 3, 4]
        self.__setup_grid()
        self.__setup_agent()

        # init after grid
        self._full_obs = self.__create_grid()
        self._base_img = self.__draw_base_img()
        self.agent_pos = {a: None for a in self.agents}
        self._step_rew = step_reward
        self._collide_rew = collide_reward

        # run env inits
        self._total_episode_reward = None
        self._agent_step_count = None
        self._agent_dones = None
        self._curr_agents_count = None

    def reset(self):
        """ Returns initial observations and states """
        self.__setup_agent()
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
            if self._step_count >= self._episode_limit:
                self._agent_dones[agent_i] = True

        return self.get_obs(), rewards, self._agent_dones

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = []
        for agent_i in self.agents:
            obs = self.get_obs_agent(agent_i)
            agents_obs.append(obs)
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._full_obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self._full_obs)

    def get_state(self):
        """ Returns the global state (all agent states) as a list """
        return self._full_obs

    def get_state_size(self):
        """ Returns the shape of the global state """
        return len(self._full_obs)

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
                fill_cell(
                    img, self.agent_pos[agent_i],
                    cell_size=CELL_SIZE, fill=AGENTS_COLORS[agent_i % AGENTS_COLORS_COUNT]
                )
                write_cell_text(
                    img, text=str(agent_i + 1),
                    pos=self.agent_pos[agent_i], cell_size=CELL_SIZE, fill='white', margin=0.3
                )
        img = np.asarray(img)
        if self._render_mode == 'rgb_array':
            return img
        elif self._render_mode == 'human':
            cv2.imshow('MARL_GRID', img)
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

    def get_episode_rew(self):
        """ Save a replay """
        return self._total_episode_reward

    def get_env_info(self):
        """ Return all key information about the environment """
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self._n_agents,
                    "episode_limit": self._episode_limit}
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
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

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
        return self.__is_valid(pos) and self._full_obs[pos[0]][pos[1]].find(PRE_IDS['agent']) > -1

    def __setup_grid(self):
        with open(self._grid_file_path, "r") as f:
            self._original_grid = [row.rstrip() for row in f.readlines()][4:]
            n_rows = len(self._original_grid)  # The number of rows in the global map
            n_cols = len(self._original_grid[0])  # The number of columns in the global map
            assert n_rows > 0 and n_cols > 0
            self._grid_shape = (n_rows, n_cols)
        return

    def __setup_agent(self):
        with open(self._agent_path, "r") as f:
            f_lines = [row.rstrip() for row in f.readlines()][1:]
            a_index = 0
            for f_line in f_lines:
                if a_index >= self._n_agents:
                    break
                line_list = f_line.replace('\t', ',').split(",")
                a_s_x, a_s_y, a_f_x, a_f_y = int(line_list[4]), int(line_list[5]), int(line_list[6]), int(line_list[7])
                self._agent_init_pos[a_index] = (a_s_x, a_s_y)
                self._agent_goal_pos[a_index] = (a_f_x, a_f_y)
                a_index += 1
        return


CELL_SIZE = 30

WALL_COLOR = 'black'

AGENTS_COLORS_COUNT = 50

AGENTS_COLORS = [
    "#F0F8FF", "#483D8B", "#79CDCD", "#00B2EE", "#1874CD", "#EE2C2C", "#EEC900", "#FF7D40", "#CD9B1D", "#27408B",
    "#FFB6C1", "#FFAEB9", "#000080", "#EECFA1", "#FFA500", "#20B2AA", "#EE00EE", "#CD8C95", "#CAE1FF", "#912CEE",
    "#FF4500", "#03A89E", "#CD69C9", "#A2B5CD", "#90EE90", "#548B54", "#DB7093", "#CD6889", "#FFEFD5", "#308014",
    "#33A1C9", "#FFBBFF", "#DDA0DD", "#FFC0CB", "#FF0000", "#8B668B", "#FFC1C1", "#87CEFF", "#71C671", "#C6E2FF",
    "#FA8072", "#F4A460", "#8B8682", "#7D9EC0", "#4EEE94", "#8B4726", "#388E8E", "#836FFF", "#4682B4", "#A602FA",
]

ACTION_MEANING = {
    0: "LEFT",
    1: "RIGHT",
    2: "UP",
    3: "DOWN",
    4: "STAY"
}

PRE_IDS = {
    'obs': 'W',
    'ept': '0',
    'agent': 'A'
}
