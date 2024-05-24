import os
from envs.marl_partial import MARL_PARTIAL_ENV
import random
import time
import numpy as np

def main():
    n_agents = 10
    grid_file_path = '/home/ubuntu/DongmingShen/MARL_curve/src/mapf_baseline/mapf-map/empty-8-8.map'
    agents_path = '/home/ubuntu/DongmingShen/MARL_curve/src/mapf_baseline/scen-random/empty-8-8-random-'
    ma_env = MARL_PARTIAL_ENV(
        grid_file_path=grid_file_path,
        agents_path=agents_path,
        n_agents=n_agents,
        episode_limit=10000,
        debug=True
    )
    print('environment resetting...')
    ma_env.reset()
    print('start testing...')
    step = 0
    while True:
        step += 1
        action = np.random.randint(5, size=n_agents)
        # print("obs:", ma_env.get_obs()[0])
        # print(action)
        obs, rew, dones = ma_env.step(action)
        av_act = ma_env.get_avail_actions()
        # print(av_act)
        # time.sleep(0.5)
        if ma_env.episode_done():
            break
        print("finished step {}".format(step))
if __name__ == "__main__":
    main()