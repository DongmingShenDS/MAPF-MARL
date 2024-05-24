import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
from envs import REGISTRY as env_REGISTRY

env = env_REGISTRY["marl_partial"](
    grid_file_path='/home/ubuntu/DongmingShen/MARL_curve/src/mapf_baseline/mapf-map/empty-8-8.map',
    agents_path='/home/ubuntu/DongmingShen/MARL_curve/src/mapf_baseline/scen-random/empty-8-8-random-',
    n_agents=10,
    obs_window=5,
    obs_knn_agents=5,  # include itself (start, goal, curr) & (unit_vec, vec_norm) & (timestep)
    episode_limit=100,
    seed=None,
    render='human',
    move_reward=-0.01,
    stay_reward=-0.02,
    stay_goal_reward=0,
    node_collide_reward=-1,
    edge_collide_reward=-1,
    env_collide_reward=-1,
    complete_reward=10,
    debug=False
)
print(env)