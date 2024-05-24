# DS DONE 1
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    """QLearner class in learners.q_learner"""
    def __init__(self, mac, scheme, logger, args):
        # DS: called by learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args) in run.py
        """init"""
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        # DS: get params => mac.parameters() from controllers/basic_controller.py => parameters() TODO
        self.params = list(mac.parameters())  # for i in self.params: print(i.size())
        # DS: SETUP MIXER if any [[[IMPORTANT]]]
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                # DS: mixer => VDNMixer() => modules.mixers.vdn
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                # DS: mixer => QMixer(args) => modules.mixers.qmix
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            # DS: add mixer.parameters() into params => modules.mixers => parameters() TODO
            self.params += list(self.mixer.parameters())
            # DS: setup target_mixer = same as mixer at start
            self.target_mixer = copy.deepcopy(self.mixer)
        # DS: setup optimiser = RMSprop => torch.optim (why RMSprop) TODO
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # DS: setup target_mac = same as mac at start
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        # DS: log_stats_t = keep track of logger output time
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # DS: batch, t_env, episode_num
        """MAIN TRAIN for QLearner"""
        "Get all relevant quantities from batch"
        # DS: rewards, actions, terminated, mask, avail_actions
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        "Calculate estimated Q-Values"
        mac_out = []
        # DS: Initialize mac hidden states => controllers/basic_controller.py => BasicMAC => mac.init_hidden(...)
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # DS: Get agent_outs, add to mac_out => controllers/basic_controller.py => BasicMAC => mac.forward(...)
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        "Calculate the Q-Values necessary for the target"
        # DS: same procedure as calculating estimated Q-Values, for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, ignore first timesteps
        # DS: Mask out unavailable actions: assign very negative value for softmax? TODO
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        "Max over target Q-Values"
        # Get actions that maximise live Q (for double q-learning); otherwise simple get target_mac_out max
        if self.args.double_q:
            # DS: get actions from mac_out max
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            # DS: get target_max_qvals from these actions & target_mac_out
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            # DS: get target_max_qvals from target_mac_out max directly
            target_max_qvals = target_mac_out.max(dim=3)[0]

        "Mix, if any mixer"
        if self.mixer is not None:
            # DS: setup mixer for chosen_action_qvals & target_max_qvals => mixers/
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        "Calculate 1-step Q-Learning targets"
        # calculate targets from reward & target_max_qvals (discounted)
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        "Td-error & L2 Loss calculation"
        # DS: calculate td_error from chosen_action_qvals & targets
        td_error = (chosen_action_qvals - targets.detach())
        # DS: expand mask to the same size as td_error
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        "Optimise"
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        "Update targets once in a while"
        # DS: => _update_targets()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        "Update logger once in a while"
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        """update targets network"""
        # DS: load current mac state into target_mac => load_state(...) in controllers/basic_controller.py
        self.target_mac.load_state(self.mac)
        # DS: load current mixer (if any) state into target_mixer => load_state_dict(...) in torch
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # DS: update logger
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        """move everything to cuda"""
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        """save models to path"""
        # DS: save mac models => save_models(path) in controllers/basic_controller.py
        self.mac.save_models(path)
        # DS: save mixer (if any) states & optimiser states
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        """load models from path"""
        # DS: load mac models => load_models(path) in controllers/basic_controller.py
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        # DS: load mixer (if any) states & optimiser states
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
