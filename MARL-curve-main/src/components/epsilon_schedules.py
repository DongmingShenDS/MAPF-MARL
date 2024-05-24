# DS DONE 1
import numpy as np


class DecayThenFlatSchedule():
    # DS: called in action_selectors
    """DecayThenFlatSchedule class in components.epsilon_schedules"""

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):
        """Init"""
        # DS: start; finish; time_length = start and finish epsilon during time_length
        self.start = start
        self.finish = finish
        self.time_length = time_length
        # DS: delta = linear/avg decay rate
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay
        # DS: setup exp_scaling if exponential decay
        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        """evaluate epsilon at time T"""
        # DS: return the result epsilon based on if the decay is linear/exp
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass
