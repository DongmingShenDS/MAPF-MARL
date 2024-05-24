import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from helloworld_v1 import HelloWorld
import random
import time
import numpy as np

map_name = 'random-64-64-20'
n_agents = 4

grid_file_path = f'../mapf_baseline/mapf-map/{map_name}.map'
agents_path = f'../mapf_baseline/scen-random/{map_name}-random-1.scen'

# ma_env = HelloWorld(
#     grid_file_path=grid_file_path,
#     agents_path=agents_path,
#     n_agents=n_agents
# )

ma_env = HelloWorld(
    n_agents=n_agents
)

ma_env.reset()
while True:
    action = np.random.randint(5, size=n_agents)
    obs, rew, dones = ma_env.step(action)
    print(obs)
    print(rew)
    # print(dones)
    exit(0)
    ma_env.render()
    if ma_env.episode_done():
        break
