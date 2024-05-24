# DS DONE 1
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# NOTE: This multi-agent controller shares parameters between agents
class BasicMAC:

    def __init__(self, scheme, groups, args):
        # DS: scheme = dict with {state, obs, actions, avail_actions, reward, terminated}
        # DS: groups = dict with {agents}
        # DS: args = command line arguments (SimpleNamespace from _config in run.py)
        """init"""
        self.n_agents = args.n_agents
        self.args = args
        # DS: input_shape=#agents; _build_agents(...) => create agent instances => agent_REGISTRY in modules/agents
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        # DS: action_selector => action_REGISTRY in components/action_selectors
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # DS: called in actions = self.mac.select_actions(...) in runners/episode_runner.py
        """Select actions. Only select actions for the selected batch elements in bs
        该方法用于在一个episode中每个时刻为所有智能体选择动作。t_ep代表当前样本在一个episode中的时间索引。t_env代表当前时刻环境运行的总时间, 用于计算epsilon-greedy中的epsilon。
        """
        # DS: get avail_actions (available actions)
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # DS: get agent_outputs from forward(ep_batch, t, test_mode)
        agent_outputs = self.forward(
            ep_batch, t_ep, test_mode=test_mode
        )
        # DS: get chosen_actions from action_selector.select_action(...) in components/action_selectors
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        # DS: inputs = ep_batch, t_ep, test_mode in select_actions(...)
        """Returns agent_outputs
        ep_batch表示一个episode的样本, t表示每个样本在该episode内的时间索引, forward()方法的作用是输出一个episode内每个时刻的观测对应的所有动作的Q值与隐层变量mac.hidden_states。
        由于QMix算法采用的是DRQN网络, 因此每个episode的样本必须与mac.hidden_states一起连续输入到神经网络中, 当t变量为0时, 即当一个episode第一个时刻的样本输入到神经网络时, 
        mac.hidden_states初始化为0, 此后在同一个episode中, mac.hidden_states会持续得到更新。
        """
        # DS: get agent_inputs from _build_inputs(ep_batch, t) for all agents
        agent_inputs = self._build_inputs(ep_batch, t)
        # DS: get avail_actions (available actions) for all agents
        avail_actions = ep_batch["avail_actions"][:, t]
        # DS: get agent_outs & hidden_state from self.agent(...) which called rnn_agent.forward()
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            # DS: apply softmax to agent_outs / agent outputs
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            # DS: when not in test_mode => pick an available action uniformly with epsilon (EPS-GREEDY)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)
                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        # DS: return agent_outs in ... format? TODO
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        """Initialize hidden states (self.hidden_states)"""
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        """Return self.agent.parameters()"""
        return self.agent.parameters()

    def load_state(self, other_mac):
        """Load self.agent's state_dict with self.agent.load_state_dict(...)"""
        # DS: load_state_dict(...) in torch
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        """Set cuda for self.agent"""
        self.agent.cuda()

    def save_models(self, path):
        """Save self.agent's state_dict() into path"""
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        """Load saved model path into self.agent"""
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        """INIT self.agent [[[IMPORTANT]]]"""
        # DS: self.agent => agent_REGISTRY => modules/agents => REGISTRY
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # DS: called in forward(...) by agent_inputs = self._build_inputs(ep_batch, t)
        """Build & return agent input
        DS: this function build all agents' network's input and return the "inputs" array
            len(inputs) = agent count
            each input in our case contains: 1. agent observation, 2. agent previous action
        """
        # Assumes homogenous agents with flat observations.
        # (other MACs might want to e.g. delegate building inputs to each agent)
        bs = batch.batch_size
        inputs = []
        # DS: input contains observation for each agent
        inputs.append(batch["obs"][:, t])  # b1av
        # DS: if obs_last_action: input contains actions
        if self.args.obs_last_action:
            if t == 0:  # DS: at t=0 (beginning), prev-actions are all 0's
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:  # DS: at t>0, prev-actions are each agent's one hot action
                inputs.append(batch["actions_onehot"][:, t - 1])
        # DS: if obs_agent_id (false in our setting): input contain ids
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        # DS: reshape & concat all inputs
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """Return input shape"""
        # DS: get observation shape from scheme => add to input_shape
        input_shape = scheme["obs"]["vshape"]
        # DS: ? TODO
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape