# DS DONE 1
import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    """VDNMixer class in mixers.vdn"""
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        """forward function"""
        # DS: VDM simply sum on each agent's Qs to get total Q-value
        return th.sum(agent_qs, dim=2, keepdim=True)