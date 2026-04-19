import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SuperSpike(torch.autograd.Function):
    """
    Spiking nonlinearity with surrogate gradient.
    Normalized negative part of a fast sigmoid (Zenke & Ganguli, 2018).
    """
    scale = 100.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad

class SQN(nn.Module):
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90, 
                 simulation_time=10, alpha=0.9, beta=0.8, threshold=1.0,
                 encoding='constant', decoding='potential'):
        super(SQN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.simulation_time = simulation_time
        
        # SNN Parameters
        self.alpha = alpha  # Synapse decay
        self.beta = beta    # Membrane decay
        self.threshold = threshold
        self.encoding = encoding # options: 'constant', 'poisson', 'equidistant'
        self.decoding = decoding # options: 'potential', 'spikes'
        self.spike_fn = SuperSpike.apply

        # Convolutional Layers (Feature Extraction)
        # Using ReLU for efficiency in the backbone, but outputting to the SNN FC layers
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Determine flattened conv feature size
        dummy_input = torch.zeros(1, *self.input_dim)
        conv_out_size = self._feature_size(dummy_input)
        self.fc_input_dim = conv_out_size + self.history_dim

        # Spiking Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, self.output_dim)

    def _feature_size(self, x):
        return self.conv(x).reshape(1, -1).size(1)

    def forward(self, state, history):
        batch_size = state.size(0)
        device = state.device

        # 1. Feature Extraction (Base features)
        features = self.conv(state)
        features_flat = features.reshape(batch_size, -1)
        
        # Combined input for the spiking FC layers
        x_fc_base = torch.cat([features_flat, history], dim=1)

        # 2. Spiking Temporal Loop
        # Initialize membrane potentials
        mem1 = torch.zeros(batch_size, 128, device=device)
        mem2 = torch.zeros(batch_size, 256, device=device)
        mem3 = torch.zeros(batch_size, self.output_dim, device=device)
        
        syn1 = torch.zeros_like(mem1)
        syn2 = torch.zeros_like(mem2)
        syn3 = torch.zeros_like(mem3)
        
        # Output trackers
        spk_count = torch.zeros(batch_size, self.output_dim, device=device)
        pot_sum = torch.zeros(batch_size, self.output_dim, device=device)

        # Equidistant counter if needed
        if self.encoding == 'equidistant':
            spike_counter = torch.ones_like(x_fc_base)
            fixed_distance = 1.0 / (x_fc_base + 1e-6)

        for t in range(self.simulation_time):
            # --- Encoding ---
            if self.encoding == 'constant':
                x_fc = x_fc_base
            elif self.encoding == 'poisson':
                spike_snapshot = torch.rand_like(x_fc_base, device=device)
                x_fc = (spike_snapshot <= x_fc_base).float()
            elif self.encoding == 'equidistant':
                x_fc = (torch.ones_like(x_fc_base) * t == torch.round(fixed_distance * spike_counter)).float()
                spike_counter += x_fc
            else:
                x_fc = x_fc_base # Default to constant

            # --- SNN Layers ---
            # FC1
            h1 = self.fc1(x_fc)
            spk1, mem1, syn1 = self._spiking_neuron(h1, mem1, syn1)
            
            # FC2
            h2 = self.fc2(spk1)
            spk2, mem2, syn2 = self._spiking_neuron(h2, mem2, syn2)
            
            # FC3 (Output Layer)
            h3 = self.fc3(spk2)
            # Update output synapse/membrane
            syn3 = self.alpha * syn3 + h3
            mem3 = self.beta * mem3 + syn3
            
            # Record potential
            pot_sum += mem3
            
            # Record spikes if needed
            if self.decoding == 'spikes':
                mthr_out = mem3 - self.threshold
                out_spk = self.spike_fn(mthr_out)
                spk_count += out_spk
                # Reset output membrane if spiking
                mem3 = mem3 - (out_spk * self.threshold)

        # --- Decoding ---
        if self.decoding == 'spikes':
            return spk_count # Total spikes in simulation time
        else: # 'potential'
            return pot_sum / self.simulation_time # Average potential

    def _spiking_neuron(self, h, mem, syn):
        # Update synapse (filter)
        new_syn = self.alpha * syn + h
        # Update membrane potential
        new_mem = self.beta * mem + new_syn
        
        # Fire spike using surrogate gradient
        mthr = new_mem - self.threshold
        out = self.spike_fn(mthr)
        
        # Reset membrane (subtraction method)
        res_mem = new_mem - (out * self.threshold)
        
        return out, res_mem, new_syn
