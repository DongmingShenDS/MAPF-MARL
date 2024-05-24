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
from utils.draw import draw_grid, fill_cell, write_cell_text  # utils.draw or draw
from smac.env.multiagentenv import MultiAgentEnv


class MAPF_GRID(MultiAgentEnv):
    """MultiAgentEnv, the base of customized envs"""

    def __init__(
            self,
            grid_file_path,
            agents_path,
            n_agents=4,
            episode_limit: int = 10000,
            seed=None,
            render='human',
            step_reward=-0.01,
            collide_reward=-10,
            debug=False
    ):
        # init
        assert os.path.exists(grid_file_path)
        self._grid_file_path = grid_file_path
        self._agent_path = agents_path
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        if seed:
            self._seed = seed
        self._render_mode = render
        self._debug_mode = debug
        # init info
        self._n_agents = n_agents
        self.agents = [a for a in range(self._n_agents)]
        self.episode_limit = episode_limit
        self._step_count = None  # environment step counter
        # init grid
        self._grid_shape = None
        self._original_grid = None
        self._agent_init_pos = {a: None for a in self.agents}
        self._agent_goal_pos = {a: None for a in self.agents}
        self._actions = [0, 1, 2, 3, 4]
        self.__setup_grid()
        self.__setup_agent()
        # init post-grid information
        self._full_obs = self.__create_grid()
        self._empty_full_obs = deepcopy(self._full_obs)
        self._map = deepcopy(self._full_obs)
        self._base_img = self.__draw_base_img()
        self.agent_positions = [(-1, -1) for _ in self.agents]
        self._step_rew = step_reward
        self._collide_rew = collide_reward
        # run env inits
        self._total_episode_reward = None
        self._agent_step_count = None
        self._agent_dones = None
        self._curr_agents_count = None

    def reset(self):
        """ Returns initial observations and states """
        self._total_episode_reward = [0 for _ in range(self._n_agents)]
        self._step_count = 0
        self._agent_step_count = [0 for _ in range(self._n_agents)]
        self._agent_dones = [False for _ in range(self._n_agents)]
        self._node_collision_agents = [0 for _ in range(self._n_agents)]
        self._edge_collision_agents = [0 for _ in range(self._n_agents)]
        self._curr_agents_count = 0
        self.agent_positions = [self._agent_init_pos[a] for a in self.agents]
        self.dir_unit_vectors = [[0, 0] for _ in self.agents]
        self.norm_unit_vectors = [0 for _ in self.agents]
        self.__init_full_obs()
        return self.get_obs()

    def step(self, agents_action):
        """ Handle Transition. Returns reward, terminated, info """
        if self._debug_mode:
            print("goals: ", self._agent_goal_pos)
        if isinstance(agents_action, torch.Tensor):
            agents_action = agents_action.detach().cpu().numpy()
        assert len(agents_action) == self._n_agents
        assert all([action_i in ACTION_MEANING.keys() for action_i in agents_action])
        self._step_count += 1  # global environment step
        rewards = [0 for _ in range(self._n_agents)]  # initialize rewards array
        new_agent_position = [self.agent_positions[agent_i] if self._agent_dones[agent_i] else None for agent_i in self.agents]
        # counting
        env_collisions = 0
        # step for every agent, ignore agents collisions first
        for agent_i, action in enumerate(agents_action):
            new_pos = self.agent_positions[agent_i]  # new_pos init to old_pos
            if not self._agent_dones[agent_i]:
                self._agent_step_count[agent_i] += 1  # agent step count
                new_pos, agent_env_flag = self.__agent_step(agent_i, action)
                new_agent_position[agent_i] = new_pos
                if agent_env_flag:
                    rewards[agent_i] += self._collide_rew
                    env_collisions += 1 
                    print("env collision")
                # assert env_collisions == 0  # update: don't allow env collisions now => change this when run actual code
                rewards[agent_i] += self._step_rew
            # checks if destination was reached
            if self.__reached_dest(agent_i, new_pos):
                self._agent_dones[agent_i] = True
                self._curr_agents_count -= 1
            # if max_steps was reached, terminate the episode
            if self._step_count >= self.episode_limit:
                self._agent_dones[agent_i] = True
            self._total_episode_reward[agent_i] += rewards[agent_i]
        # detect node collisions
        node_collisions, self._node_collision_agents = self.__count_node_collision(new_agent_position)
        # detect edge collisions
        edge_collisions, self._edge_collision_agents = self.__count_edge_collision(new_agent_position)
        if self._debug_mode:
            print("node_collisions_agents: ", self._node_collision_agents)
            print("edge_collisions_agents: ", self._edge_collision_agents)
        # update rewards based on collision information
        for agent_i in self.agents:
            rewards[agent_i] += (self._collide_rew * self._node_collision_agents[agent_i])  # node collision
            rewards[agent_i] += (self._collide_rew * self._edge_collision_agents[agent_i])  # edge collision
            self._total_episode_reward[agent_i] += rewards[agent_i]
        # update agent's possition and full_obs
        self._full_obs = deepcopy(self._empty_full_obs)
        for agent_i in self.agents:
            self.agent_positions[agent_i] = new_agent_position[agent_i]
            self.__update_agent_view(agent_i)
        # info stores the info about this transition
        info = {}
        info['_step_count'] = self._step_count
        # update available actions after this step
        self._avail_actions = [self.get_avail_agent_actions(agent_i) for agent_i in self.agents]
        return sum(rewards), self._agent_dones, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        self.__update_goal_vectors()  # update vector observation
        if self._debug_mode:
            print("observation dict")
            obs_dict = {
                "map": np.array(self._full_obs),
                "positions": np.array(self.agent_positions),
                "goals": np.array(self.agent_goals),
                "starts": np.array(self.agent_starts),
                "directions": np.array(self.dir_unit_vectors),
                "distances": np.array(self.norm_unit_vectors),
                "timesteps": np.array(self._agent_step_count),
                "node_collision": np.array(self._node_collision_agents),
                "edge_collision": np.array(self._edge_collision_agents)
            }
            for k, v in obs_dict.items():
                print(k, ':\n', v)
        return np.array([self.get_obs_agent(agent_i) for agent_i in self.agents])

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs = np.concatenate(
            (
                np.array(self._full_obs), # full map: size_map
                np.array(self.agent_positions), # agents' positions: 2*N
                np.array(self.agent_goals), # agents' goals: 2*N
                np.array(self.agent_starts), # agents' initial positions: 2*N
                np.array(self.dir_unit_vectors), # unit vec to goal: 2*N
                np.array(self.norm_unit_vectors), # vec norm to goal: N
                np.array(self._agent_step_count), # timestep: N
                np.array(self._node_collision_agents), # node collision: N
                np.array(self._edge_collision_agents) # edge collision: N
            ), axis=None
        ) # axis=None to flatten otherwise episode_buffer raise error
        obs = np.concatenate(
            (
                np.array(self._full_obs), # full map: size_map
            ), axis=None
        ) # axis=None to flatten otherwise episode_buffer raise error
        return obs  

    def get_obs_size(self):
        # obs size for each agent
        # return self._grid_shape[0] * self._grid_shape[1] + 12 * self._n_agents
        return self._grid_shape[0] * self._grid_shape[1]

    def get_state(self):
        """ Returns the global state (all agent states) as a list """
        return np.array(self._full_obs).flatten()  # flatten otherwise episode_buffer raise error

    def get_state_size(self):
        """ Returns the shape of the global state """
        return self._grid_shape[0] * self._grid_shape[1]  # state size for the environment

    def get_avail_actions(self):
        """ Returns the available actions of all agents in a list """
        self._avail_actions = [self.get_avail_agent_actions(agent_i) for agent_i in self.agents]
        return self._avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid_actions = [0, 0, 0, 0, 0]
        curr_pos = self.agent_positions[agent_id]
        # left / action 0
        left = (curr_pos[0] - 1, curr_pos[1])
        if self.__is_valid(left) and not self.__is_cell_obstacle(left):
            valid_actions[0] = 1
        # right / action 1
        right = (curr_pos[0] + 1, curr_pos[1])
        if self.__is_valid(right) and not self.__is_cell_obstacle(right):
            valid_actions[1] = 1
        # up / action 2
        up = (curr_pos[0], curr_pos[1] - 1)
        if self.__is_valid(up) and not self.__is_cell_obstacle(up):
            valid_actions[2] = 1
        # down / action 3 
        down = (curr_pos[0], curr_pos[1] + 1)
        if self.__is_valid(down) and not self.__is_cell_obstacle(down):
            valid_actions[3] = 1
        valid_actions[4] = 1  # stay / action 4 is always allowed
        return valid_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return len(self._actions)

    def render(self):
        """ Render something when have a monitor """
        img = copy.copy(self._base_img)
        for agent_i in range(self._n_agents):
            if not self._agent_dones[agent_i]:
                fill_cell(img, self.agent_positions[agent_i], cell_size=CELL_SIZE, fill=AGENTS_COLORS[agent_i])
                write_cell_text(img, text=str(agent_i + 1), pos=self.agent_positions[agent_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)
        img = np.asarray(img)
        if self._render_mode == 'rgb_array':
            return img
        elif self._render_mode == 'human':
            cv2.imshow('GridWorld', img)
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
        self._full_obs[self.agent_positions[agent_i][0]][self.agent_positions[agent_i][1]] += 1

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

    def __reached_dest(self, agent_i, new_pos):
        # verifies if the agent_i reached a destination place, if reach, leave agent there
        if new_pos == self._agent_goal_pos[agent_i]:
            return True
        return False

    def __agent_step(self, agent_i, action):
        # return an agent's new position based on action
        # return new_pos & flag: if collide with environment (True if collision happens)
        curr_pos = copy.copy(self.agent_positions[agent_i])
        if action == 0:  # LEFT
            next_pos = (curr_pos[0] - 1, curr_pos[1])
        elif action == 1:  # RIGHT
            next_pos = (curr_pos[0] + 1, curr_pos[1])
        elif action == 2:  # UP
            next_pos = (curr_pos[0], curr_pos[1] - 1)
        elif action == 3:  # DOWN
            next_pos = (curr_pos[0], curr_pos[1] + 1)
        elif action == 4:  # STAY
            return curr_pos, False
        else:
            raise Exception('Action Not found!')
        # if there is an agent out of range collision
        if not self.__is_valid(next_pos):
            return curr_pos, True
        # if there is an agent-obstacle collision
        if self.__is_cell_obstacle(next_pos):
            return curr_pos, True
        # no agent-env collision, return normally
        return next_pos, False

    def __count_node_collision(self, new_agent_positions):
        if self._debug_mode:
            print("agent prev positions:", self.agent_positions)
            print("agent new positions:", new_agent_positions)
        agent_at_grid = dict()
        collision_agents = [0 for _ in self.agents]
        collisions_count = 0
        for agent_i, pos in enumerate(new_agent_positions):
            if pos not in agent_at_grid:  # new pos
                agent_at_grid[pos] = {agent_i}
            else:  # existing pos
                agent_at_grid[pos].add(agent_i)
        for pos, agents in agent_at_grid.items():
            count = len(agents)
            if count > 1:  # more than 1 agents at 1 grid
                collisions_count += count
                for agent_i in agents:
                    collision_agents[agent_i] += 1
        return collisions_count, collision_agents

    def __count_edge_collision(self, new_agent_positions):
        old_agent_positions = np.array([self.__index_to_number(pos) for pos in self.agent_positions])
        new_agent_positions = np.array([self.__index_to_number(pos) for pos in new_agent_positions])
        collision_agents = [0 for _ in self.agents]
        collisions_count = 0
        for agent_i, pos in enumerate(new_agent_positions):
            i_old_pos = old_agent_positions[agent_i]
            if i_old_pos == pos:
                continue  # i's position not change => not possible edge conflict
            agent_js = np.where(old_agent_positions == pos)[0]  # j_old==i_new (pos)
            if len(agent_js) == 0:
                continue
            for agent_j in agent_js:
                if agent_j == agent_i:
                    continue
                j_new_pos = new_agent_positions[agent_j]
                if j_new_pos == i_old_pos and j_new_pos != pos:  # j_new==i_old and j_new!=j_old
                    collisions_count += 1
                    collision_agents[agent_i] += 1
        return collisions_count, collision_agents

    def __update_agent_pos(self, agent_i, action):
        # updates the agent position in the environment
        curr_pos = copy.copy(self.agent_positions[agent_i])
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
            self.agent_positions[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['ept']
            self.__update_agent_view(agent_i)
        return False

    def __check_collision(self, pos):
        # verifies if a transition to the position pos will result on a collision.
        # return self.__is_valid(pos) and self._full_obs[pos[0]][pos[1]].find(PRE_IDS['agent']) > -1
        return self.__is_valid(pos) and self._full_obs[pos[0]][pos[1]] > 0

    def __setup_grid(self):
        with open(self._grid_file_path, "r") as f:
            self._original_grid = [row.rstrip() for row in f.readlines()][4:]
            n_rows = len(self._original_grid)  # The number of rows in the global map
            n_cols = len(self._original_grid[0])  # The number of columns in the global map
            assert n_rows > 0 and n_cols > 0
            self._grid_shape = (n_rows, n_cols)
        return

    def __setup_agent(self):
        # randomly sample a scen file (index between 1 and 25)
        random_scen_path = self._agent_path + str(random.randint(1, 25)) + '.scen'
        assert os.path.exists(random_scen_path)
        with open(random_scen_path, "r") as f:
            f_lines = [row.rstrip() for row in f.readlines()][1:]
            # randomly sample _n_agents lines from the file with agent info
            sampled_lines = random.sample(f_lines, self._n_agents)
            a_index = 0
            for f_line in sampled_lines:
                if a_index >= self._n_agents:
                    break
                line_list = f_line.replace('\t', ',').split(",")
                a_s_x, a_s_y, a_f_x, a_f_y = int(line_list[4]), int(line_list[5]), int(line_list[6]), int(line_list[7])
                self._agent_init_pos[a_index] = (a_s_x, a_s_y)
                self._agent_goal_pos[a_index] = (a_f_x, a_f_y)
                self.agent_starts = [self._agent_init_pos[i] for i in range(self._n_agents)]
                self.agent_goals = [self._agent_goal_pos[i] for i in range(self._n_agents)]
                a_index += 1
        return

    def __update_goal_vectors(self):
        # update vectors of (agent_pos -> agent_goal)
        for i in range(self._n_agents):
            pos, goal = self.agent_positions[i], self.agent_goals[i]
            distance = [goal[0] - pos[0], goal[1] - pos[1]]
            norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
            if norm == 0:
                direction = [0, 0]
            else:
                direction = [distance[0] / norm, distance[1] / norm]
            self.dir_unit_vectors[i] = direction
            self.norm_unit_vectors[i] = norm
        return

    def __index_to_number(self, index):
        # 1-to-1 map each (x, y) to a single number by y * num_row + x
        assert len(index) == 2
        return self._grid_shape[0] * index[1] + index[0]


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
    4: "STAY",
}

PRE_IDS = {
    'obs': -1,
    'ept': 0,
    'agent': ''
}
