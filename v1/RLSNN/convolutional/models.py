import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import snntorch as snn
from snntorch import surrogate

class ConvDQN(nn.Module):
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90):
        super(ConvDQN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_input_dim = self.feature_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state, history):
        features = self.conv(state)
        features = features.reshape(features.size(0), -1)
        features = torch.cat([features, history], dim=1)
        q_values = self.fc(features)
        return q_values

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_dim)).reshape(1, -1).size(1) + self.history_dim

class LIFNeuron(nn.Module):
    def __init__(self, tau=2.0, v_threshold=1.0):
        super(LIFNeuron, self).__init__()
        self.decay = 1.0 - (1.0 / tau)
        self.v_t = v_threshold
        self.memp = 0.0 

    def reset(self):
        self.memp = 0.0

    def forward(self, x):
        self.memp = self.memp * self.decay + x

        spike = (self.memp >= self.v_t).float()

        self.memp = self.memp - spike * self.v_t
        return spike

class SpikingWrapperV1(nn.Module):
    def __init__(self, model, num_steps=100):
        super(SpikingWrapper, self).__init__()
        self.num_steps = num_steps
        
        self.conv1 = model.conv[0]
        self.conv2 = model.conv[2]
        self.conv3 = model.conv[4]
        
        self.fc1 = model.fc[0]
        self.fc2 = model.fc[2]
        self.fc3 = model.fc[4]

    def multi_spike_if(self, current, mem, threshold=1.0):
        if mem is None:
            mem = torch.zeros_like(current)
        mem = mem + current
        # Output exactly how many discrete threshold crossings occurred (prevents >1.0 activation saturation)
        spk = torch.floor(F.relu(mem) / threshold)
        mem = mem - spk * threshold
        return spk, mem

    def forward(self, state, history):
        state_size = state.size(0)
        
        mem1 = None
        mem2 = None
        mem3 = None
        mem4 = None
        mem5 = None
        
        out_V = None

        for step in range(self.num_steps):
            # Direct constant-current encoding is safer for image frames than pure Poisson variance
            spk_in = state

            c1 = self.conv1(spk_in)
            spk1, mem1 = self.multi_spike_if(c1, mem1)
            
            c2 = self.conv2(spk1)
            spk2, mem2 = self.multi_spike_if(c2, mem2)
            
            c3 = self.conv3(spk2)
            spk3, mem3 = self.multi_spike_if(c3, mem3)
            
            spk3_flat = spk3.reshape(state_size, -1)
            
            # Sub-features inject naturally into the stream
            x = torch.cat([spk3_flat, history.float()], dim=1)
            
            f1 = self.fc1(x)
            spk4, mem4 = self.multi_spike_if(f1, mem4)
            
            f2 = self.fc2(spk4)
            spk5, mem5 = self.multi_spike_if(f2, mem5)
            
            out = self.fc3(spk5)

            if out_V is None:
                out_V = out
            else:
                out_V += out
                
        # Average continuous output across the whole time window to get steady Q-values
        return out_V / self.num_steps


class SpikingWrapperV2(nn.Module):
    def __init__(self, model, num_steps=100):
        super(SpikingWrapper, self).__init__()
        self.num_steps = num_steps
        
        self.conv1 = model.conv[0]
        self.conv2 = model.conv[2]
        self.conv3 = model.conv[4]
        
        self.fc1 = model.fc[0]
        self.fc2 = model.fc[2]
        self.fc3 = model.fc[4]

        self.lif = LIFNeuron()

    def forward(self, state, history):
        state_size = state.size(0)
        self.lif.reset()
        out_V = None

        for step in range(self.num_steps):
            spk_in = state

            c1 = self.conv1(spk_in)
            spk1 = self.lif(c1)
            
            c2 = self.conv2(spk1)
            spk2 = self.lif(c2)
            
            c3 = self.conv3(spk2)
            spk3 = self.lif(c3)
            
            spk3_flat = spk3.reshape(state_size, -1)
            
            x = torch.cat([spk3_flat, history.float()], dim=1)
            
            f1 = self.fc1(x)
            spk4 = self.lif(f1)
            
            f2 = self.fc2(spk4)
            spk5 = self.lif(f2)
            
            out = self.fc3(spk5)

            if out_V is None:
                out_V = out
            else:
                out_V += out
                
        return out_V / self.num_steps