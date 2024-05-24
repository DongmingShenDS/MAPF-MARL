# DS DONE 1
import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule


class MultinomialActionSelector():
    """MultinomialActionSelector class in components.action_selectors"""

    def __init__(self, args):
        """Init"""
        self.args = args
        # DS: schedule => DecayThenFlatSchedule in components/epsilon_schedules
        self.schedule = DecayThenFlatSchedule(args.epsilon_start,
                                              args.epsilon_finish,
                                              args.epsilon_anneal_time,
                                              decay="linear")
        # DS: epsilon at 0 => schedule.eval(...) => eval() in components/epsilon_schedules
        self.epsilon = self.schedule.eval(0)
        # DS: test_greedy: if testing = using pure greedy
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """Select next step actions at t_env"""
        # DS: setup masked_policies, mask excluded actions (avail_actions==0.0) to 0.0 ? TODO
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0
        # DS: get the epsilon at t_env from schedule.eval(t_env) => eval() in components/epsilon_schedules
        self.epsilon = self.schedule.eval(t_env)
        # DS: get & return picked_actions
        if test_mode and self.test_greedy:
            # DS: if testing (test_mode & test_greedy), select action using pure greedy
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            # DS: if not testing, sample action from masked_policies using Categorical(...).sample() in torch
            picked_actions = Categorical(masked_policies).sample().long()
        return picked_actions


class EpsilonGreedyActionSelector():
    """EpsilonGreedyActionSelector class in components.action_selectors"""

    def __init__(self, args):
        """Init"""
        self.args = args
        # DS: schedule => DecayThenFlatSchedule in components/epsilon_schedules
        self.schedule = DecayThenFlatSchedule(args.epsilon_start,
                                              args.epsilon_finish,
                                              args.epsilon_anneal_time,
                                              decay="linear")
        # DS: epsilon at 0 => schedule.eval(...) => eval() in components/epsilon_schedules
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # called in BasicMAC.select_actions() in controllers/basic_controller
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        """Select next step actions at t_env"""
        # DS: get the epsilon at t_env from schedule.eval(t_env) => eval() in components/epsilon_schedules
        self.epsilon = self.schedule.eval(t_env)
        if test_mode:
            # DS: if testing (test_mode), greedy action selection only
            self.epsilon = 0.0
        # DS: setup masked_policies: mask excluded actions (avail_actions==0.0) from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        # DS: pick actions with epsilon-greedy for each actions (vectorized?) TODO
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


# DS: called in self.action_selector = action_REGISTRY(...) in controllers/basic_controller.py
REGISTRY = {}
REGISTRY["multinomial"] = MultinomialActionSelector
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
