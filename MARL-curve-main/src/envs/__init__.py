# DS DONE 1
"""This file setup any customized multiagentenv"""
from functools import partial
import socket
import sys
import os
# do not import SC2 in labtop TODO: server error
from .multiagentenv import MultiAgentEnv
# if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname():
#     from smac.env import MultiAgentEnv, StarCraft2Env
# else:
#     from .multiagentenv import MultiAgentEnv

# from .stag_hunt import StagHunt
# from .GridworldEnv import GridworldEnv
# from .GridworldEnv2 import GridworldEnv2
# from .GridworldEnv3 import GridworldEnv3
# from .GridworldEnv4 import GridworldEnv4
# from .GridworldEnv5 import GridworldEnv5
# from .GridworldEnvNew import GridworldEnvnew
# from .Pushbox import PushBox
# from .aloha import AlohaEnv
# from .mapf_gridworld import MAPF_GRID
from .marl_partial import MARL_PARTIAL_ENV

# from .pursuit import PursuitEnv
# from .sensors import SensorEnv
# from .hallway import HallwayEnv
# from .disperse import DisperseEnv
# from .gather import GatherEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    # print(env)
    # for x in kwargs:
    #     print(x)
    return env(**kwargs)

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# REGISTRY = {
#     "sc2": partial(env_fn, env=StarCraft2Env),
#     # "matrix_game_1": partial(env_fn, env=Matrix_game1Env),
#     # "matrix_game_2": partial(env_fn, env=Matrix_game2Env),
#     # "matrix_game_3": partial(env_fn, env=Matrix_game3Env),
#     # "mmdp_game_1": partial(env_fn, env=mmdp_game1Env),
#     # "mmdp_game_2": partial(env_fn, env=TwoState),
#     # "spread_x": partial(env_fn, env=spread_xEnv),
#     # "spread_x2": partial(env_fn, env=spread_x2Env),
# } if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname() else {}
# REGISTRY["gridworld"] = GridworldEnv
# REGISTRY["gridworld2"] = GridworldEnv2
# REGISTRY["gridworld3"] = GridworldEnv3
# REGISTRY["gridworld4"] = GridworldEnv4
# REGISTRY["gridworld5"] = GridworldEnv5
# REGISTRY["gridworldnew"] = GridworldEnvnew
# REGISTRY["pushbox"] = PushBox
# REGISTRY["aloha"] = partial(env_fn, env=AlohaEnv)
# REGISTRY["mapf_gridworld"] = partial(env_fn, env=MAPF_GRID)
REGISTRY["marl_partial"] = partial(env_fn, env=MARL_PARTIAL_ENV)

# REGISTRY["pursuit"] = partial(env_fn, env=PursuitEnv)
# REGISTRY["sensor"] = partial(env_fn, env=SensorEnv)
# REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)
# REGISTRY["disperse"] = partial(env_fn, env=DisperseEnv)
# REGISTRY["gather"] = partial(env_fn, env=GatherEnv)
# REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))