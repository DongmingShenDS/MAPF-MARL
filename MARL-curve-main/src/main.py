# DS DONE 1
import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
# DS: Sacred is a tool to configure, organize, log and reproduce computational experiments.
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

# DS: Some of Sacred’s general behaviour is configurable via sacred.SETTINGS
# DS: https://sacred.readthedocs.io/en/stable/settings.html
SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console

# DS: utils.logging -> get_logger(): set up a custom logger
logger = get_logger()

# DS: sacred_Experiment is the central class of the Sacred framework (create, run, configure)
# DS: https://sacred.readthedocs.io/en/stable/experiment.html
ex = Experiment("pymarl")

# DS: customize the logging behaviour of your experiment by just providing a custom Logger object to your experiment
ex.logger = logger

# DS: captured_out_filter = Filter function to be applied to captured output of a run
# DS: apply_backspaces_and_linefeeds =To interpret control characters like a console this would do
ex.captured_out_filter = apply_backspaces_and_linefeeds

# DS: create result folder
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    # run the framework
    # DS: when will this get called? TODO
    # DS: what is _run, _config, _log? TODO
    print("run(_run, config, _log) in main.py called")
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    # DS: params = system command line arguments params (as list)
    # DS: arg_name = one particular argument
    # DS: sub-folder under config/ (algs or envs)
    config_name = None
    for _i, _v in enumerate(params):
        # DS: if arg_name specified in params, save to config_name
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break
    # DS: when config_name exits, load specific alg/env .yaml configurations from the specified yaml file
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                # DS: load contents in yaml into a dict = config_dict
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    # DS: d = original config_dict
    # DS: u = extra config_dict'
    # DS: goal is to update u into d (if DNE, add; if E, update)
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':

    # DS: load command line argv (as list)
    params = deepcopy(sys.argv)

    # DS: default.yaml = default hyper-parameters loading into config_dict
    # config_file = "customize.yaml"
    config_file = input("Enter config file yaml: \n")  # test.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", config_file), "r") as f:
        try:
            # DS: load contents in yaml into a dict = config_dict
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "{} error: {}".format(config_file, exc)

    # DS: Load algorithm and env base configs => _get_config(params, arg_name, subfolder)
    env_config = _get_config(params, "--env-config", "envs")    # DS: environment configuration
    alg_config = _get_config(params, "--config", "algs")        # DS: algorithm configuration
    # DS: update all extra configuration into config_dict as a dict => recursive_dict_update
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # DS: Configuration entries can also directly be added as a dictionary using the ex.add_config method
    # DS: https://sacred.readthedocs.io/en/stable/configuration.html
    ex.add_config(config_dict)

    # DS: Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")

    # DS: Experiments in Sacred collect lots of information about runs. Use observer interface to access the info.
    # DS: https://sacred.readthedocs.io/en/stable/experiment.html?highlight=Experiment%20observers#observe-an-experiment
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # DS: Run the command-line interface of this experiment.
    # DS: https://sacred.readthedocs.io/en/stable/apidoc.html?highlight=run_commandline
    ex.run_commandline(params)

