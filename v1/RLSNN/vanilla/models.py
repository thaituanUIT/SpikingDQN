import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import RLSNN.vanilla.parameters as parameters
import snntorch as snn
from snntorch import surrogate

class SpikingDQNv1(nn.Module):
    def __init__(self):
        super(SpikingDQN, self).__init__()

        self.num_steps = 10
        self.linear1 = nn.Linear((4096 + parameters.action_option*parameters.history_size), 1024)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1, spike_grad=surrogate.atan())
        self.linear2 = nn.Linear(1024, 1024)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1, spike_grad=surrogate.atan())
        self.linear3 = nn.Linear(1024, 9)

    def forward(self, x):
        if hasattr(self.lif1, 'threshold') and isinstance(self.lif1.threshold, torch.Tensor):
            self.lif1.threshold = self.lif1.threshold.to(x.device)
        if hasattr(self.lif2, 'threshold') and isinstance(self.lif2.threshold, torch.Tensor):
            self.lif2.threshold = self.lif2.threshold.to(x.device)

        mem1 = None
        mem2 = None
        
        spk2_record = []

        for _ in range(self.num_steps):
            l1 = self.linear1(x)
            spk1, mem1 = self.lif1(l1, mem1)
            l2 = self.linear2(spk1)
            spk2, mem2 = self.lif2(l2, mem2)
            spk2_record.append(spk2)

        spk2_stacked = torch.stack(spk2_record)
        spk_sum = spk2_stacked.sum(dim=0)
        q_values = self.linear3(spk_sum)
        return q_values

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear((4096 + parameters.action_option*parameters.history_size), 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 9)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        q_values = self.linear3(x)
        return q_values
