# DS DONE 1
from envs import REGISTRY as env_REGISTRY
from functools import partial
# DS: components/episode_buffer.py => EpisodeBatch(...)
from components.episode_buffer import EpisodeBatch
import numpy as np
TEST_OUT = False  # Global to control if print test output

class EpisodeRunner:
    # DS: called in runner=r_REGISTRY... in run_sequential(args, logger) in run.py
    """EpisodeRunner class in runner.episode_runner"""

    def __init__(self, args, logger):
        """Init"""
        self.args = args
        self.logger = logger
        # DS: episode runner: batch_size==1
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        # DS: env => envs => REGISTRY as env_REGISTRY in envs/init.py => [[[IMPORTANT]]] to pick env
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        # DS: init others
        self.t = 0
        self.t_env = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        # DS: called in runner.setup(...) in run.py
        """Setup new_batch & mac"""
        # DS: partial? https://docs.python.org/3/library/functools.html TODO
        # DS: when call new_batch, equivalent to call EpisodeBatch(...) in components/episode_buffer.py
        self.new_batch = partial(EpisodeBatch,
                                 scheme,
                                 groups,
                                 self.batch_size,
                                 self.episode_limit + 1,
                                 preprocess=preprocess,
                                 device=self.args.device)
        # DS: mac => BasicMAC in controllers/basic_controller.py
        self.mac = mac

    def get_env_info(self):
        """Get env information => call env.get_env_info()"""
        return self.env.get_env_info()

    def save_replay(self):
        """Save replay => call env.save_replay()"""
        self.env.save_replay()

    def close_env(self):
        """Close the env => call env.close()"""
        self.env.close()

    def reset(self):
        """Reset env => update batch & call env.reset()"""
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        # DS: called in runner.run(...) in run.py, test_mode=False when testing
        """Run env for a whole episode, can define test_mode (affect action selection)"""
        # DS: reset batch & env => reset()
        self.reset()
        terminated = False
        episode_return = 0
        # DS: init hidden states => mac.init_hidden(...) in BasicMAC in controllers/basic_controller
        self.mac.init_hidden(batch_size=self.batch_size)
        # DS: keep running until terminated==True
        while not terminated:
            # DS: get info before transition
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            # DS: update contents in batch (EpisodeBatch) => .update(...) in components/episode_buffer.EpisodeBatch
            self.batch.update(pre_transition_data, ts=self.t)
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # DS: get action => mac.select_actions(...) in BasicMAC in controllers/basic_controller
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # DS: perform transition => env.step(...) in envs/multiagentenv
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            # DS: get info after transition
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],  # DS: if episode_limit not in env_info => False
            }
            # DS: update contents in batch (EpisodeBatch) => .update(...) in components/episode_buffer.EpisodeBatch
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1
        # DS: get info of final step
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        
        if test_mode and TEST_OUT:  # DS
            # from IPython import embed; embed()  # breakpoint
            print("TEST_OUT: ")  # print something below for reference
        # DS: update contents in batch (EpisodeBatch) => .update(...) in components/episode_buffer.EpisodeBatch
        self.batch.update(last_data, ts=self.t)
        # DS: select actions in the last stored state & update batch (EpisodeBatch)
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        # DS: get final info ? TODO
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)
        if not test_mode:
            self.t_env += self.t
        cur_returns.append(episode_return)
        # DS: update logger, make log once in a while by calling _log(...)
        # DS: note that _log(...) will clear cur_returns & cur_stats
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        # DS: return the batch (EpisodeBatch)
        # from IPython import embed; embed()
        return self.batch

    def _log(self, returns, stats, prefix):
        """Update log, display useful information"""
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
