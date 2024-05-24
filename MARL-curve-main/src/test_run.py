# from run.py to test
import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):
    """main RUN function called by main.py
    run.py 文件中run函数的主要作用是构建实验参数变量 args 以及一个自定义Logger类的记录器 logger
    构建好 args 变量和 logger 对象后, 就通过 run_sequential(args, logger) 进入到了run.py文件中的 run_sequential 函数中

    args变量:
        内置变量_config的拷贝作为参数传入到了run函数中, _config 是字典变量, 因此查看参数时, 需要利用 _config[key]=value, 
        在 run 函数中, 作者构建了一个namespace类的变量args, 将_config中的参数都传给了 args, 这样就可以通过args.key=value的方式查看参数了。

    logger对象:
        ex 对象的内置变量作为参数传入到了 run(_run, config, _log) 函数中, 此时我们可以将实验结果记录在 _run.info 变量中, 
        同时可以利用 _log.info() 在控制台打印一些实验过程中的中间信息。
        为了更好地利用这两个功能, 在 run 函数中, 作者定义了一个自定义utils.logging.Logger类的对象 logger
            将 _log 赋给了 logger.console_logger, 
            将 _run.info 赋给 logger.sacred_info,

    然后定义了Logger类的两个重要方法:
    - logger.log_stat(key, value, t, to_sacred=True) 方法: 
        将实验过程中产生的结果定期记录在 logger.sacred_info 变量中，其中 key 是键值, value 是结果值, t 表示索引；
    - logger.print_recent_stats() 方法: 
        用于定期在控制台打印实验结果。
    """
    # DS: check args sanity and possibly update wrong configs (or quit)
    _config = args_sanity_check(_config, _log)
    # DS: SN = type.SimpleNamespace = can instantiate an object that can hold attributes and nothing else (holder)
    # DS: args = _config's contents into an extractable class
    args = SN(**_config)
    # DS: set up args.device according to args.use_cuda
    args.device = "cuda" if args.use_cuda else "cpu"

    # DS: setup customized loggers = utils.logging.Logger
    logger = Logger(_log)
    _log.info("Experiment Parameters:")
    # DS: setup Experiment Parameters in some format
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # DS: configure Tensorboard logger if args.use_tensorboard
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # DS: sacred is on in logger by default ? TODO
    logger.setup_sacred(_run)

    """ DS: MAIN RUN_SEQUENTIAL (run and train) => run_sequential(args, logger) """
    print("Enter RUN_SEQUENTIAL")
    run_sequential(args=args, logger=logger)

    # DS: Clean up after finishing
    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)  # DS: terminate thread t
            print("Thread joined")
    print("Exiting script")

    # DS: Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    print("INSIDE evaluate_sequential, test_nepisode = {}".format(args.test_nepisode))
    sum_eps_rew = 0
    sum_eps_collide_count = 0
    sum_eps_len = 0
    sum_eps_total_cost = 0
    for i in range(args.test_nepisode):
        print("TEST {}".format(i))
        # get test_buffer (with testing information) back from runner.run
        test_buffer = runner.run(test_mode=True)
        # from IPython import embed; embed()  # breakpoint - see test_buffer
        # discounted_r = 0
        # for r in test_buffer['reward'].detach().cpu().numpy()[0]:
        #     discounted_r = 0.99 * discounted_r + r
        # print("reward =", th.sum(test_buffer['reward']).detach().cpu().numpy())
        # print("discounted reward =", discounted_r)
        sum_eps_rew += th.sum(test_buffer['reward']).detach().cpu().numpy()
        sum_eps_collide_count += max(test_buffer['state'].detach().cpu().numpy()[0][:, 0])
        sum_eps_len += max(test_buffer['state'].detach().cpu().numpy()[0][:, 1])
        sum_eps_total_cost += max(test_buffer['state'].detach().cpu().numpy()[0][:, 2])
    print("evaluate_sequential result summary for {} runs: ".format(args.test_nepisode))
    print("\tavg eps rew = {}".format(sum_eps_rew / args.test_nepisode))
    print("\tavg eps collide = {}".format(sum_eps_collide_count / args.test_nepisode))
    print("\tavg eps length = {}".format(sum_eps_len / args.test_nepisode))
    print("\tavg eps total cost = {}".format(sum_eps_total_cost / args.test_nepisode))
    if args.save_replay:
        runner.save_replay()
    runner.close_env()


def run_sequential(args, logger):
    """DS: run_sequential(args, logger) == the MAIN TRAINING function
    run_sequential 是实验运行的主要函数, 作用是首先是构建如下自定义类的对象: 
        EpisodeRunner类的环境运行器对象 runner, 
        ReplayBuffer类的经验回放池对象 buffer,
        BasicMAC 类的智能体控制器对象 mac,
        以及QLeaner类的智能体学习器对象 leaner,
    然后进行实验，即训练智能体，记录实验结果，定期测试并保存模型。
    """

    """RUNNER: runner对象 (环境运行器)
    runner对象属于自定义的runner.episode_runner.EpisodeRunner类, 该对象的主要作用是运行环境以产生训练样本, 
    因此runner对象中的一个重要属性就是env.multiagentenv.MultiAgentEnv类的环境对象runner.env, 即环境, 
    而另一个属性是components.episode_buffer.EpisodeBatch类的episode样本存储器对象runner.batch, 该对象用于以episode为单位存储环境运行所产生的样本。

    runner对象中最关键的方法是:
    - runner.run(test_mode=False): 
        利用当前智能体mac在环境中运行(需要用到mac对象), 产生一个episode的样本数据episode_batch, 存储在runner.batch中。
    """
    # DS: r_REGISTRY => runners.REGISTRY in runners/init.py
    # DS: args.runner should be episode (EpisodeRunner) or parallel (ParallelRunner)
    # DS: init the runner (EpisodeRunner/ParallelRunner) with args & logger
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    "Set up schemes and groups here"
    # DS: env_info = get env info from runner => get_env_info() in runners/
    env_info = runner.get_env_info()
    # DS: set up env_info n_agents, n_actions, state_shape
    # DS: these should be setup in multiagentenv.py => MultiAgentEnv
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    # Default/Base scheme
    # DS: what is a scheme? - just some dictionary with information
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # DS: what is a group? TODO
    groups = {
        "agents": args.n_agents
    }
    # DS: OneHot => components/transforms.py => OneHot(...) of actions
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    """BUFFER: buffer对象 (经验回放池)
    buffer对象属于自定义的components.episode_buffer.ReplayBuffer(EpisodeBatch)类, 该对象的主要作用是存储样本以及采样样本。
    ReplayBuffer的父类是EpisodeBatch。EpisodeBatch类对象用于存储episode的样本, ReplayBuffer(EpisodeBatch)类对象则用于存储所有的off-policy样本, 
    也即EpisodeBatch类变量的样本会持续地补充到ReplayBuffer(EpisodeBatch)类的变量中。
    同样由于QMix用的是DRQN结构, 因此EpisodeBatch与ReplayBuffer中的样本都是以episode为单位存储的。
    在EpisodeBatch中数据的维度是[batch_size, max_seq_length, *shape], ReplayBuffer类数据的维度是[buffer_size, max_seq_length, *shape]。
        EpisodeBatch中Batch Size表示此时batch中有多少episode, 
        ReplayBuffer中episodes_in_buffer表示此时buffer中有多少个episode的有效样本。
        max_seq_length则表示一个episode的最大长度。

    buffer对象中的关键方法有:
    - buffer.insert_episode_batch(ep_batch): 
        将EpisodeBatch类变量ep_batch中的样本全部存储到buffer中。
    - buffer.sample(batch_size): 
        从buffer中取出batch_size个episode的样本用于训练, 这些样本组成了EpisodeBatch类的对象。
    """
    # DS: buffer => components/episode_buffer.py => ReplayBuffer(...)
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # from IPython import embed; embed()  # breakpoint
    
    """MAC: mac对象 (智能体控制器)
    mac对象属于自定义的controller.basic_controller.BasicMAC类, 该对象的主要作用是控制智能体, 
    因此mac对象中的一个重要属性就是nn.module类的智能体对象mac.agent, 该对象定义了各个智能体的局部Q网络, 即接收观测作为输入, 输出智能体各个动作的Q值。
    另外, QMix使用的是DRQN结构, 因此mac.agent用的是RNN网络, mac.hidden_states储存了RNN的隐层变量。

    mac对象有两个关键方法: 
    - mac.forward(ep_batch, t, test_mode=False): 
        ep_batch表示一个episode的样本, 
        t表示每个样本在该episode内的时间索引, 
        forward()方法的作用是输出一个episode内每个时刻的观测对应的所有动作的Q值与隐层变量mac.hidden_states。
        由于QMix算法采用的是DRQN网络, 因此每个episode的样本必须与mac.hidden_states一起连续输入到神经网络中, 
        当t变量为0时, 即当一个episode第一个时刻的样本输入到神经网络时, mac.hidden_states初始化为0。此后在同一个episode中, mac.hidden_states会持续得到更新。
    - mac.select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        该方法用于在一个episode中每个时刻为所有智能体选择动作。
        t_ep代表当前样本在一个episode中的时间索引。
        t_env代表当前时刻环境运行的总时间, 用于计算epsilon-greedy中的epsilon。
    """
    # Setup multi-agent controller here
    # DS: mac => controllers => REGISTRY as mac_REGISTRY in controllers/init.py
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    """RUNNER: runner对象 (环境运行器)
    runner对象属于自定义的runner.episode_runner.EpisodeRunner类, 该对象的主要作用是运行环境以产生训练样本, 
    因此runner对象中的一个重要属性就是env.multiagentenv.MultiAgentEnv类的环境对象runner.env, 即环境, 
    而另一个属性是components.episode_buffer.EpisodeBatch类的episode样本存储器对象runner.batch, 该对象用于以episode为单位存储环境运行所产生的样本。
    
    runner对象中最关键的方法是:
    - runner.run(test_mode=False): 
        利用当前智能体mac在环境中运行 (需要用到mac对象), 产生一个episode的样本数据episode_batch, 存储在runner.batch中。
    """
    # DS: runner.setup(scheme, groups, preprocess, mac) in runners/
    # DS: pass everything needed into runner with .setup(...) == initialization
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    """LEARNER: 
    learner对象属于自定义的leaners.q_learner.QLearner(与具体选择哪个算法有关), 该对象的主要作用是依据特定算法对智能体参数进行训练更新。
    在QMix算法与VDN算法中, 均有nn.module类的混合网络learner.mixer, 
    因此learner对象需要学习的参数包括各个智能体的局部Q网络参数mac.parameters(), 以及混合网络参数learner.mixer.parameters(), 两者共同组成了learner.params, 然后用优化器learner.optimiser进行优化。
    
    learner对象中的关键方法是:
    - learner.train(batch: EpisodeBatch, t_env: int, episode_num: int): 
        batch表示当前用于训练的样本, t_env表示当前环境运行的总时间步数, episode_num表示当前环境运行的总episode数, 该方法利用特定算法对learner.params进行更新。
    """
    # DS: learner => learners => REGISTRY as le_REGISTRY in learners/init.py
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    "CHECKPOINT IF ANY"
    # DS: Load args.checkpoint_path for trained policy (if any)
    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0
        # DS: Check if args.checkpoint_path actually exists, if not: exit
        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return
        # DS: Extract timesteps from the checkpoint_path into a [list]
        # go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
        # DS: args.load_step==0 then load the max timestep, otherwise load the timestep closest to args.load_step
        if args.load_step == 0:
            timestep_to_load = max(timesteps)  # choose the max timestep
        else:
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))  # choose the closest to load_step
        # DS: extract model path
        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        # DS: logger update
        logger.console_logger.info("Loading model from {}".format(model_path))
        # DS: learner.load_models() => in learners/
        learner.load_models(model_path)
        # DS: runner.t_env (what is t_env? timestep?) => in runners/ TODO
        runner.t_env = timestep_to_load
        # DS: if args.evaluate, args.save_replay => evaluate_sequential(args, runner) TODO
        if args.evaluate or args.save_replay:
            print("EVALUATE SEQUENTIAL")
            evaluate_sequential(args, runner)
            return

    "START TESTING WITHOUT TRAINING"
    print("START TESTING WITHOUT TRAINING")
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    # DS: logger update, time update
    start_time = time.time()
    last_time = start_time
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    # DS: runner.t_env=current timestep, args.t_max=max timestep to terminate
    while runner.t_env <= args.t_max:
        # "Run for a whole episode at a time"
        # # DS: runner.run(...) in runners/ = run the environment for an episode
        # episode_batch = runner.run(test_mode=False)
        # # DS: ReplayBuffer.insert_episode_batch(...) in components/episode_buffer.py = insert episode into buffer
        # buffer.insert_episode_batch(episode_batch)
        # # DS: ReplayBuffer.sample(...) = sample a batch from the episode if can
        # if buffer.can_sample(args.batch_size):
        #     episode_sample = buffer.sample(args.batch_size)
        #     # Truncate batch to only filled timesteps
        #     # DS: all in components/episode_buffer.py, ? TODO
        #     max_ep_t = episode_sample.max_t_filled()
        #     episode_sample = episode_sample[:, :max_ep_t]
        #     if episode_sample.device != args.device:
        #         episode_sample.to(args.device)
        #     # DS: learner.train(...) in learners/ => TRAIN this episode
        #     learner.train(episode_sample, runner.t_env, episode)

        "Execute test runs once in a while"
        # DS: n_test_runs = number of test runs? TODO
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        # DS: when to perform test
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            # DS: logger update, time update
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            last_test_T = runner.t_env
            # DS: runner.run(...) in runners/ = run the environment for n_test_runs episodes for TESTING
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # "Save model once in a while"
        # if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
        #     model_save_time = runner.t_env
        #     # DS: save_path = path to the saved model
        #     save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        #     os.makedirs(save_path, exist_ok=True)
        #     # DS: logger update
        #     logger.console_logger.info("Saving models to {}".format(save_path))
        #     # learner should handle saving/loading -- delegate actor save/load to mac,
        #     # use appropriate filenames to do critics, optimizer states
        #     # DS: learner.save_model(...) in learners/ => SAVE current model to save_path
        #     learner.save_models(save_path)

        "Post-processing for every episode"
        # DS: args.batch_size_run? TODO
        episode += args.batch_size_run
        # DS: handle logger once in a while according to args.log_interval
        if (runner.t_env - last_log_T) >= args.log_interval:
            # DS: logger update with logger.log_stat(...) => in utils/logging.py
            logger.log_stat("episode", episode, runner.t_env)
            # DS: logger output/print to console with logger.print_recent_stats(...) => in utils/logging.py
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # DS: runner.close_env() in runners/ = manually close environment when finished
    runner.close_env()
    # DS: logger update
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    """check & fix/update (if possible) command line args = configs, update log"""
    # DS: set CUDA flags
    # DS: Use cuda whenever possible!
    config["use_cuda"] = True
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")
    # DS: set test_nepisode according to batch_size_run
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]
    # DS: return the checked (updated) config
    return config
