# DS DONE 1
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """RNNAgent class in modules.agents.rnn_agent"""
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        # DS: get parameters used => input_shape, args.rnn_hidden_dim, args.n_actions
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        """Init hidden state (fc1) for an agent"""
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """Define forward() for the Neural Net"""
        # input here are observation + available actions for each agent
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
