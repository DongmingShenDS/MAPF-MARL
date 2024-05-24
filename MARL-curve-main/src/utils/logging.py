# DS DONE 1
from collections import defaultdict
import logging
import numpy as np
import torch as th


class Logger:
    """Logger class in utils.logging"""
    def __init__(self, console_logger):
        # DS: console_logger = _log = system console logger pointer? TODO
        self.console_logger = console_logger
        # DS: default = no use tb (tensorboard), sacred (sacred), hdf (?)
        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        # DS: stats? TODO
        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        """Set up Tensorboard logger"""
        # import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        """Set up sacred (on in logger by default)"""
        # DS: sacred_run_dict = ? TODO
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        """Update log stat"""
        # DS: append new information to states
        self.stats[key].append((t, value))
        # DS: update to tensorboard
        if self.use_tb:
            self.tb_logger(key, value, t)
        # DS: update to sacred_info (sacred information)
        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        """Print out (recent) status"""
        # DS: setup log information to print
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            # DS: error: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu()...  
            item = "{:.4f}".format(np.mean([x[1].cpu() if th.is_tensor(x[1]) else x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n"
            # log_str += "\n" if i % 4 == 0 else "\t"  # TODO changed by DS
        # DS: logger update/print (to console)
        self.console_logger.info(log_str)


def get_logger():
    """Set up a custom logger"""
    # DS: getLogger() from logging package = create custom logger instance
    logger = logging.getLogger()
    # DS: set up logger information ... TODO
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')
    return logger

