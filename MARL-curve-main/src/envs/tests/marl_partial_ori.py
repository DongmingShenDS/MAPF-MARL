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
from utils.draw import draw_grid, fill_cell, write_cell_text    # utils.draw or draw
from smac.env.multiagentenv import MultiAgentEnv
# from IPython import embed; embed()                            # breakpoint


class MARL_PARTIAL_ENV(MultiAgentEnv):
    """MultiAgentEnv, the base of customized envs"""

    def __init__(
            self,
            grid_file_path,
            agents_path,
            n_agents=4,
            obs_window=5,
            obs_knn_agents=5,  # include itself (start, goal, curr) & (unit_vec, vec_norm) & (timestep)
            episode_limit=200,
            seed=None,
            render='human',
            move_reward=-0.01,
            stay_reward=-0.02,
            stay_goal_reward=0,
            node_collide_reward=-1,
            edge_collide_reward=-1,
            env_collide_reward=-1,
            complete_reward=30,
            debug=False
    ):
        """
        Initialize the MARL_PARTIAL_ENV->MultiAgentEnv environment
        """
        "init"
        # print("INIT CALLED")
        assert os.path.exists(grid_file_path)
        self._grid_file_path = grid_file_path       # mapf file path
        self._agent_path = agents_path              # partial mapf agent start & goal path
        self._render_mode = render                  # render mode (doesn't metter in server)
        self._debug_mode = debug                    # print debug info
        self._n_agents = n_agents                   # number of agents = N
        self._seed = random.randint(0, 9999)        # random seed
        np.random.seed(self._seed)
        if seed:
            self._seed = seed

        "agent init"
        self.agents = [a for a in range(self._n_agents)]                            # all agents' index (from 0)
        self.agents_pairs = []                                                      # actual agent pairs (for distance updata)
        self.agent_distance_matrix = np.zeros((self._n_agents, self._n_agents))     # NxN agent's distance matrix
        self._agent_init_pos = [(-1, -1) for _ in self.agents]                      # agent initial positions
        self._agent_goal_pos = [(-1, -1) for _ in self.agents]                      # agent goal positions
        self._agent_positions = [(-1, -1) for _ in self.agents]                     # agent current positions
        for a1 in self.agents:
            for a2 in self.agents:
                if (a1, a2) in self.agents_pairs or (a2, a1) in self.agents_pairs:  # or a1==a2?
                    continue
                self.agents_pairs.append((a1, a2))
        assert len(self.agents_pairs) == self._n_agents * (self._n_agents - 1) / 2 + self._n_agents
        self._n_features = 13                   # number of features (numbers) for each agent
        self._actions = [0, 1, 2, 3, 4]         # agent available actions
        self.episode_limit = episode_limit      # max episode len allowed (should not be too big for RAM issue)
        self._step_count = None                 # environment step counter
        self._move_rew = move_reward            # reword/punishment for L, R, U, D 4 direction move
        self._stay_rew = stay_reward            # reward/punishment for no move (stay/S) & not at goal
        self._stay_goal_rew = stay_goal_reward  # reward/punishment for no move (stay/S) & at goal
        self._nc_rew = node_collide_reward      # reward/punishment for (agent) node collision
        self._ec_rew = edge_collide_reward      # reward/punishment for (agent) edge collision
        self._env_c_rew = env_collide_reward    # reward/punishment for environment collision
        self._complete_rew = complete_reward    # reward/punishment for successfully complete the episode
        self._obs_knn_agents = obs_knn_agents   # how many nearest agents to observe, default 5, if not enough set to 0
        self._obs_window = obs_window           # W: how big space/view window to observe, agent centered
        self._window_shape = (obs_window, obs_window)

        "grid init"
        self._grid_shape = None                 # grid shape, usually LxL square
        self._original_grid = None              # stores the raw ascii grid from .map file
        self.__setup_grid()                     # setup map/grid's initial information
        self.__setup_agent()                    # setup agent's initial information (random) TODO: where to put this?
        self._full_obs = self.__create_grid()   # the LxL full map observation -1 for obs, 0 for empty, n for #agents
        self._base_img = self.__draw_base_img()
        self._empty_full_obs = deepcopy(self._full_obs)

        "env/helper/other init"
        self._total_episode_reward = None   # total reward for an episode for each agent
        self._agent_step_count = None       # number of steps for each agent, if finished, stop counting
        self._agent_at_goals = None         # if agent is at goal for each agent
        self._agent_dones = None            # if agent finish for each agent
        self._curr_agents_count = None      # current number of agent not yet finish the goal TODO: where update?
        self._node_collision_agents = None  # (N) array for node collisions, i=1 means agent i collide
        self._edge_collision_agents = None  # (N) array for edge collisions, i=1 means agent i collide
        self._dir_unit_vectors = None       # (N) direction unit vector from each agent's current to its goal
        self._norm_unit_vectors = None      # (N) distance from each agent's current to its goal
        self._avail_actions = None          # (5) available actions, each agent has one such vector

    def reset(self):
        """
        Returns initial observations (and states?)
        """
        "randomly setup agent when reset, then update everything else"
        self.__setup_agent()
        self._full_obs = self.__create_grid()
        self._base_img = self.__draw_base_img()
        self._empty_full_obs = deepcopy(self._full_obs)

        "init environment variables"
        self._terminated = False
        self._step_count = 0                                                    # _step_count init to 0
        self._total_episode_reward = [0 for _ in self.agents]                   # _total_episode_reward init to 0's
        self._agent_step_count = [0 for _ in self.agents]                       # _agent_step_count init to 0's
        self._agent_at_goals = [False for _ in self.agents]                     # _agent_at_goals init to F's
        self._agent_dones = [False for _ in self.agents]                        # _agent_dones init to F's
        self._node_collision_agents = [0 for _ in self.agents]                  # _node_collision_agents init to 0's
        self._edge_collision_agents = [0 for _ in self.agents]                  # _edge_collision_agents init to 0's
        self._curr_agents_count = self._n_agents                                # _curr_agents_count init to _n_agents
        self._agent_positions = [self._agent_init_pos[a] for a in self.agents]  # position init to _agent_init_pos
        self._dir_unit_vectors = [[0, 0] for _ in self.agents]                  # _dir_unit_vectors init to [0, 0]'s
        self._norm_unit_vectors = [0 for _ in self.agents]                      # _norm_unit_vectors init to 0's
        self._avail_actions = [0, 0, 0, 0, 0]                                   # _avail_actions init to 0's

        "setup environment observations"
        # instantiate full observation (called __update_agent_view)
        self.__init_full_obs()
        self._empty_full_obs = deepcopy(self._full_obs)
        # update _dir_unit_vectors & _norm_unit_vectors
        self.__update_goal_vectors()
        # update available actions for the next(first) step
        self._avail_actions = [self.__get_avail_agent_actions(agent_i) for agent_i in self.agents]
        # update agent_distance_matrix (update all non-diag/unequal pairs, diag stays 0)
        self.__update_distance_matrix()
        return self.get_obs()

    def step(self, agents_action):
        """
        Handle Transition/Action. Returns reward, terminated, info
        """
        # sanity check
        if isinstance(agents_action, torch.Tensor):
            agents_action = agents_action.detach().cpu().numpy()
        assert len(agents_action) == self._n_agents
        assert all([action_i in ACTION_MEANING.keys() for action_i in agents_action])
        # update global environment step counter
        self._step_count += 1
        if self._debug_mode:
            print("current step: ", self._step_count)
            print("\tagent actions: ", agents_action)
            print("\tgoals: ", self._agent_goal_pos)
        # initialize rewards array (N): 0 for each agent
        rewards = [0 for _ in range(self._n_agents)]
        # new_agent_positions stores the temp position after transition: take action blindly
        new_agent_positions = [self.agent_pos(agent_i) for agent_i in self.agents]
        env_collisions = 0

        "take action for every agent, ignoring any agent collisions for now"
        for agent_i, action in enumerate(agents_action):
            # init new_pos to old_pos before transition
            new_pos = self.agent_pos(agent_i)
            # step all non-finished agents
            if not self._agent_dones[agent_i]:
                self._agent_step_count[agent_i] += 1
                # self.__agent_step(...) handles a single agent's single step, returns new_pos & agent_env_flag
                new_pos, agent_env_flag = self.__agent_step(agent_i, action)
                new_agent_positions[agent_i] = new_pos
                # agent_env_flag is True if collide with environment, in training this should never happen
                if agent_env_flag:
                    rewards[agent_i] += self._env_c_rew
                    env_collisions += 1
                # handle transition cost: move / stay has different rewards
                if action in {0, 1, 2, 3}:
                    rewards[agent_i] += self._move_rew
                elif action == 4:
                    if self._agent_at_goals[agent_i]:
                        rewards[agent_i] += self._stay_goal_rew
                    else:
                        rewards[agent_i] += self._stay_rew
                else:
                    assert action in self._actions
            # __reached_dest(...) checks if this agent's destination was reached at new_pos
            self._agent_at_goals[agent_i] = False
            if self.__reached_dest(agent_i, new_pos):
                # if reach goal, set agent goal reached & reduce agent count
                self._agent_at_goals[agent_i] = True
            # check if this agent's max_steps was reached, if so terminate the agent's episode
            if self._step_count >= self.episode_limit:
                self._terminated = True
                self._agent_dones[agent_i] = True
                self._curr_agents_count -= 1

        "check & handle agent collisions"
        if self._debug_mode:
            print("\tagent prev positions:", self._agent_positions)
            print("\tagent new positions:", new_agent_positions)
        # __check_node_collisions(...) detects node collisions for all agents after taking action
        node_collisions, self._node_collision_agents = self.__check_node_collisions(new_agent_positions)
        # __check_edge_collisions(...) detects edge collisions for all agents after taking action
        edge_collisions, self._edge_collision_agents = self.__check_edge_collisions(new_agent_positions)
        if self._debug_mode:
            print("\tnode_collisions_agents: ", self._node_collision_agents)
            print("\tedge_collisions_agents: ", self._edge_collision_agents)
        # update rewards based on the collision information (after this step, rewards for each agent are complete)
        for agent_i in self.agents:
            # if agent_i has node collision, then self._node_collision_agents[agent_i]=1, otherwise=0
            rewards[agent_i] += (self._nc_rew * self._node_collision_agents[agent_i])
            # if agent_i has edge collision, then self._node_collision_agents[agent_i]=1, otherwise=0
            rewards[agent_i] += (self._ec_rew * self._edge_collision_agents[agent_i])
        # info stores the info about this transition TODO: what usage?
        info = {'_step_count': self._step_count}

        "keep everything updated after each step/transition/action"
        # update _agent_positions, _full_obs, _total_episode_reward
        self._full_obs = deepcopy(self._empty_full_obs)
        for agent_i in self.agents:
            self._agent_positions[agent_i] = new_agent_positions[agent_i]
            self.__update_agent_view(agent_i)
            self._total_episode_reward[agent_i] += rewards[agent_i]
        # update _dir_unit_vectors & _norm_unit_vectors
        self.__update_goal_vectors()
        # update available actions for the next step
        self._avail_actions = [self.__get_avail_agent_actions(agent_i) for agent_i in self.agents]
        # update agent_distance_matrix (update all non-diag/unequal pairs, diag stays 0)
        self.__update_distance_matrix()
        # update _agent_dones: finish if all goal reached
        if sum(self._agent_at_goals) == self._n_agents:
            self._agent_dones = [True for _ in range(self._n_agents)]
            self._terminated = True
        if self._debug_mode:
            print("\tagent at goals: ", self._agent_at_goals)
            print("\tagent rewards: ", rewards)
        return sum(rewards), self._terminated, info

    def get_obs(self):
        """
        Returns all agents' observations in a list
        """
        # return an array of each agent's observation
        return np.array([self.get_obs_agent(agent_i) for agent_i in self.agents])

    def get_obs_agent(self, agent_id):
        """
        Returns observation for a single agent: agent_id
        """
        "agent_id's partially observation window WxW size matrix"
        # top_left_obs is the top-left corner index of agent_id's observation window
        assert agent_id > -1
        top_left_obs = (self.agent_pos(agent_id)[0] - self._obs_window // 2,
                        self.agent_pos(agent_id)[1] - self._obs_window // 2)
        # obstacle_map (WxW): 1 (obstacle or out of bound), 0 (otherwise)
        obstacle_map = np.zeros(self._window_shape)
        # agents_map (WxW): >0 (number of agents), 0 (otherwise)
        agents_map = np.zeros(self._window_shape)
        for i in range(top_left_obs[0], top_left_obs[0] + self._obs_window):
            for j in range(top_left_obs[1], top_left_obs[1] + self._obs_window):
                # if not valid (out-of-map-bound), same as obstacles, set to obstacle_map
                if not self.__is_valid((i, j)):
                    obstacle_map[i - top_left_obs[0], j - top_left_obs[1]] = 1
                    continue
                # otherwise, (i, j) is a valid index in the map, observe based on _full_obs
                if self._full_obs[i][j] == -1:  # obstacle
                    obstacle_map[i - top_left_obs[0], j - top_left_obs[1]] = 1
                elif self._full_obs[i][j] > 0:  # number of agents
                    agents_map[i - top_left_obs[0], j - top_left_obs[1]] = self._full_obs[i][j]

        "agent_id's K-nearest-neighbor agents' features observation (including itself always at first!!!)"
        # obs_knn stores each knn agent's features, ordered from nearest to farthest, fill not enough with -1
        obs_knn = np.zeros((self._obs_knn_agents, self._n_features)) - 1
        # agent_distance are each agent's distance (ordered by index, not sorted) to agent_id
        agents_distance = self.agent_distance_matrix[agent_id]
        # if _n_agents < _obs_knn_agents (not enough), we select top-_n_agents, otherwise select top-_obs_knn_agents
        k_m1 = min(self._n_agents, self._obs_knn_agents) - 1
        # knn_agents stores knn agents' index in ascending distance order, first is itself, break tie randomly
        knn_agents = sorted(range(len(agents_distance)), key=lambda sub: agents_distance[sub])[:k_m1]
        knn_agents.insert(0, agent_id)
        # update obs_knn with these knn_agents's info
        for i, na_id in enumerate(knn_agents):
            curr = self.agent_pos(na_id)                    # (2) current location
            start = self._agent_init_pos[na_id]             # (2) initial location
            goal = self._agent_goal_pos[na_id]              # (2) goal location
            unit_vec = self._dir_unit_vectors[na_id]        # (2) unit vec direction to goal
            dist_vec = self._norm_unit_vectors[na_id]       # (1) L2 distance to goal
            node_c = self._node_collision_agents[na_id]     # (1) if node collision
            edge_c = self._edge_collision_agents[na_id]     # (1) if edge collision
            agent_dist = agents_distance[na_id]             # (1) L2 distance with the agent
            timestep = self._agent_step_count[na_id]        # (1) timestep taken until now
            obs_knn[i] = np.array([
                curr[0], curr[1],
                start[0], start[1],
                goal[0], goal[1],
                unit_vec[0], unit_vec[1],
                dist_vec, node_c, edge_c, agent_dist, timestep
            ])

        "combine/flatten everything & return"
        obs = np.concatenate((obstacle_map, agents_map, obs_knn), axis=None)
        return obs

    def get_obs_size(self):
        """
        Returns the shape of the observation (obs size for each agent)
        """
        obs_size = 2 * (self._obs_window ** 2) + self._obs_knn_agents * self._n_features
        return obs_size

    def get_state(self):
        """
        Returns the global state (all agent states) as a list
        """
        # we don't use global state
        return np.array([0])

    def get_state_size(self):
        """
        Returns the shape of the global state
        """
        # we don't use global state
        return 1

    def get_avail_actions(self):
        """
        Returns the available actions of all agents in a list (_n_agents * 5)
        """
        # return _avail_actions directly, must always keep _avail_actions updated (reset & step)
        return self._avail_actions

    def __get_avail_agent_actions(self, agent_i):
        """
        Returns the available actions for agent_id in a (5) boolean list: 1 means ith action available 0 means not
        """
        # init valid_actions to all False(invalid)
        valid_actions = [0, 0, 0, 0, 0]
        curr_pos = self.agent_pos(agent_i)
        # Left: if left position is valid and not obstacle => set left/action 0 to True(valid)
        left = (curr_pos[0] - 1, curr_pos[1])
        if self.__is_valid(left) and not self.__is_cell_obstacle(left):
            valid_actions[0] = 1
        # Right: if right position is valid and not obstacle => set right/action 1 to True(valid)
        right = (curr_pos[0] + 1, curr_pos[1])
        if self.__is_valid(right) and not self.__is_cell_obstacle(right):
            valid_actions[1] = 1
        # Up: if up position is valid and not obstacle => set up/action 2 to True(valid)
        up = (curr_pos[0], curr_pos[1] - 1)
        if self.__is_valid(up) and not self.__is_cell_obstacle(up):
            valid_actions[2] = 1
        # Down: if down position is valid and not obstacle => set down/action 3 to True(valid)
        down = (curr_pos[0], curr_pos[1] + 1)
        if self.__is_valid(down) and not self.__is_cell_obstacle(down):
            valid_actions[3] = 1
        # Stay: stay/action 4 is always True(valid)
        valid_actions[4] = 1
        return valid_actions

    def get_total_actions(self):
        """
        Returns the total number of actions an agent could ever take
        """
        return len(self._actions)

    def render(self):
        """
        Render something when have a monitor, not care in AWS
        """
        img = copy.copy(self._base_img)
        for agent_i in range(self._n_agents):
            if not self._agent_dones[agent_i]:
                pos = self.agent_pos(agent_i)
                fill_cell(
                    img, pos, cell_size=CELL_SIZE, fill=AGENTS_COLORS[agent_i]
                )
                write_cell_text(
                    img, text=str(agent_i + 1), pos=pos, cell_size=CELL_SIZE, fill='white', margin=0.3
                )
        img = np.asarray(img)
        if self._render_mode == 'human':
            cv2.imshow('MARL', img)
            cv2.waitKey(1)
        return img

    def close(self):
        """
        Close the environment
        """
        pass

    def seed(self):
        """
        Set seed for the environment
        """
        pass

    def save_replay(self):
        """
        Save a replay
        """
        pass

    def get_env_info(self):
        """
        Return all key information about the environment
        """
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self._n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def episode_done(self):
        """
        Checks if episode is complete (iff all agents are done)
        """
        return sum(self._agent_dones) == self._n_agents

    def agent_pos(self, agent_id):
        """
        Returns agent_id's current position
        """
        assert -1 < agent_id < self._n_agents
        return self._agent_positions[agent_id]

    def __is_valid(self, pos):
        """
        Checks if pos is valid (inside the grid range)
        """
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def __is_cell_vacant(self, pos):
        """
        Checks check if pos is available (safe) to move an agent into
        """
        return self.__is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['ept'])

    def __is_cell_obstacle(self, pos):
        """
        Checks if pos is an obstacle
        """
        return self.__is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['obs'])

    def __create_grid(self):
        """
        Creates empty grid = to be used as _full_obs
        """
        _grid = [[PRE_IDS['ept'] for _ in range(self._grid_shape[1])] for _ in range(self._grid_shape[0])]
        for i in range(self._grid_shape[1]):
            for j in range(self._grid_shape[0]):
                _grid[i][j] = PRE_IDS['ept'] if self._original_grid[i][j] == '.' else PRE_IDS['obs']
        return _grid

    def __init_full_obs(self):
        """
        Inits _full_obs (puts agents into _full_obs) & Updates _base_img
        """
        # Initiates environment, put agents in the env
        self._full_obs = self.__create_grid()
        for agent_i in self.agents:
            self.__update_agent_view(agent_i)
        self._base_img = self.__draw_base_img()

    def __update_distance_matrix(self):
        """
        Updates the agent_distance_matrix with the new agent positions
        """
        # this must assume agent positions are initialized (reset) & updated (step) accordingly
        for (a_id1, a_id2) in self.agents_pairs:
            # assert a_id1 != a_id2  # in our setting, there are pairs with same elements
            distance = self.__get_l2_distance(a_id1, a_id2)
            self.agent_distance_matrix[a_id1][a_id2] = distance
            self.agent_distance_matrix[a_id2][a_id1] = distance
        for a_id in self.agents:
            self.agent_distance_matrix[a_id][a_id] = self.__get_l2_distance(a_id, a_id)

    def __get_l2_distance(self, a_id1, a_id2):
        """
        Gets L2 distance between two agents
        """
        # distance with itself assigned to a big number to avoid being picked in KNN selection
        if a_id1 == a_id2:
            return self._grid_shape[0] * self._grid_shape[1]
        (x1, y1) = self.agent_pos(a_id1)
        (x2, y2) = self.agent_pos(a_id2)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __get_l1_distance(self, a_id1, a_id2):
        """
        Gets L1 distance between two agents
        """
        # distance with itself assigned to a big number to avoid being picked in KNN selection
        if a_id1 == a_id2:
            return self._grid_shape[0] * self._grid_shape[1]
        (x1, y1) = self.agent_pos(a_id1)
        (x2, y2) = self.agent_pos(a_id2)
        return abs(x1 - x2) + abs(y1 - y2)

    def __update_agent_view(self, agent_i):
        """
        Updates agent_i's view in _full_obs (add the count there by 1)
        """
        # put agent_i in the _full_obs
        pos = self.agent_pos(agent_i)
        self._full_obs[pos[0]][pos[1]] += 1

    def __draw_base_img(self):
        """
        Draw the image
        """
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
        """
        Checks if agent_i reaches its goal at new_pos
        """
        # verifies if the agent_i's new pos == its goal
        if new_pos == self._agent_goal_pos[agent_i]:
            return True
        return False

    def __agent_step(self, agent_i, action):
        """
        Handles agent_i's transition by taking action. Returns its new position & env_collision_flag
        """
        # set up the agent's new position based on action
        curr_pos = copy.copy(self.agent_pos(agent_i))
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
        # if there is an agent out of range collision, return original position & True
        if not self.__is_valid(next_pos):
            return curr_pos, True
        # if there is an agent-obstacle collision, return original position & True
        if self.__is_cell_obstacle(next_pos):
            return curr_pos, True
        # no agent-env collision, return new position & False
        return next_pos, False

    def __check_node_collisions(self, new_positions):
        """
        Detects node collisions for all agents after step. Returns count & node collision indicator vector
        """
        # agents_at_pos_dict: map from a pos to {set of agents at pos}
        agents_at_pos_dict = dict()
        # collision_agents stores node collision indicator vector: if ith term is 1 then agent_i has node collisions
        collision_agents = [0 for _ in self.agents]
        collisions_count = 0
        # for each agent's and its new pos, update agents_at_pos_dict
        for agent_i, pos in enumerate(new_positions):
            if pos not in agents_at_pos_dict:       # new pos: init set at [pos]
                agents_at_pos_dict[pos] = {agent_i}
            else:                                   # existing pos: add to set at [pos]
                agents_at_pos_dict[pos].add(agent_i)
        # for each pos-agents pair in agents_at_pos_dict, check number of agents at pos
        for pos, agents_at_pos in agents_at_pos_dict.items():
            count = len(agents_at_pos)
            # if > 1 agents at 1 pos, then they collide, update collisions_count & collision_agents accordingly
            if count > 1:
                collisions_count += count
                for agent_i in agents_at_pos:
                    collision_agents[agent_i] += 1
        assert sum(collision_agents) == collisions_count
        return collisions_count, collision_agents

    def __check_edge_collisions(self, new_positions):
        """
        Detects edge collisions for all agents after step. Returns count & edge collision indicator vector
        """
        # old_positions & new_positions stores the old & new 1-to-1 INT position of agents, respectively
        old_positions = np.array([self.__index_to_number(pos) for pos in self._agent_positions])
        new_positions = np.array([self.__index_to_number(pos) for pos in new_positions])
        # collision_agents stores edge collision indicator vector: if ith term is 1 then agent_i has edge collisions
        collision_agents = [0 for _ in self.agents]
        collisions_count = 0
        # for each agent_i's and its new_pos, check its old_pos
        for agent_i, i_new_pos in enumerate(new_positions):
            i_old_pos = old_positions[agent_i]
            # if position not change, then not possible edge conflict, continue
            if i_old_pos == i_new_pos:
                continue
            # get agents that were (old position) at agent_i's new_pos; j_old_pos==i_new_pos
            agent_js = np.where(old_positions == i_new_pos)[0]
            # if no agents were at agent_i's new_pos, then not possible edge conflict, continue
            if len(agent_js) == 0:
                continue
            # otherwise, for each of such agent_j: check agent_j's new_pos
            for agent_j in agent_js:
                if agent_j == agent_i:
                    continue
                j_new_pos = new_positions[agent_j]
                # if j's new_pos == i's old_pos (in addition to j's old_pos == i's new_pos), then they swapped
                if j_new_pos == i_old_pos:
                    assert j_new_pos != i_new_pos  # TODO: delete later after testing
                    # swapped means edge conflict, update collisions_count & collision_agents accordingly
                    collisions_count += 1
                    collision_agents[agent_i] += 1
        assert sum(collision_agents) == collisions_count
        return collisions_count, collision_agents

    def __check_collision(self, pos):
        """
        Check collision at position: check if pos has agent-node-collision
        """
        # verifies if pos currently has a collision
        return self.__is_valid(pos) and self._full_obs[pos[0]][pos[1]] > 0

    def __setup_grid(self):
        """
        Initializes the map from the .map file
        """
        with open(self._grid_file_path, "r") as f:
            # _original_grid stores the ascii grid map
            self._original_grid = [row.rstrip() for row in f.readlines()][4:]
            n_rows = len(self._original_grid)       # number of rows in the global map
            n_cols = len(self._original_grid[0])    # number of columns in the global map
            self._grid_shape = (n_rows, n_cols)     # _grid_shape init to (#rows, #cols) of the grid map
            assert n_rows > 0 and n_cols > 0
        return

    def __setup_agent(self):
        """
        Initializes agent start & goal positions from the .scen file randomly
        """
        # randomly sample a scen file (index between 1 and 25) corresponds to the map
        random_scen_path = self._agent_path + str(random.randint(1, 25)) + '.scen'
        if self._debug_mode:
            print(f"scen file used: {random_scen_path}")
        assert os.path.exists(random_scen_path)
        with open(random_scen_path, "r") as f:
            f_lines = [row.rstrip() for row in f.readlines()][1:]
            # randomly sample _n_agents lines from the file
            assert len(f_lines) > self._n_agents
            rand_lines = random.sample(f_lines, self._n_agents)
            a_index = 0
            for f_line in rand_lines:
                if a_index >= self._n_agents:
                    break
                line_list = f_line.replace('\t', ',').split(",")
                a_s_x, a_s_y, a_f_x, a_f_y = int(line_list[4]), int(line_list[5]), int(line_list[6]), int(line_list[7])
                # set up each agent's initial position & goal position
                self._agent_init_pos[a_index] = (a_s_x, a_s_y)
                self._agent_goal_pos[a_index] = (a_f_x, a_f_y)
                a_index += 1
        return

    def __update_goal_vectors(self):
        """
        Updates _dir_unit_vectors & _norm_unit_vectors for all agents
        """
        # update vectors of current -> goal for each agent
        for i in range(self._n_agents):
            pos, goal = self.agent_pos(i), self._agent_goal_pos[i]
            distance = [goal[0] - pos[0], goal[1] - pos[1]]
            norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
            if norm == 0:
                direction = [0, 0]
            else:
                direction = [distance[0] / norm, distance[1] / norm]
            self._dir_unit_vectors[i] = direction
            self._norm_unit_vectors[i] = norm
        return

    def __index_to_number(self, index):
        """
        Maps: 1-to-1 map each (x, y) to a single number by y * num_row + x
        """
        assert len(index) == 2
        return self._grid_shape[0] * index[1] + index[0]


CELL_SIZE = 30

WALL_COLOR = 'black'

AGENTS_COLORS = [
    "red",
    "blue",
    "yellow",
    "orange",
    "green",
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
    'agent': '',
}