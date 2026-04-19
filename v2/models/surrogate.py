"""Implementation for SNN with Surrogate Gradient learning"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class SuperSpike(torch.autograd.Function):
    """
    Spiking nonlinearity with surrogate gradient (SuperSpike).
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

class SQNSurrogate(nn.Module):
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90, 
                 simulation_time=10, alpha=0.9, beta=0.8, threshold=1.0, use_vgg16=False):
        super(SQNSurrogate, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.simulation_time = simulation_time
        self.use_vgg16 = use_vgg16

        
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.spike_fn = SuperSpike.apply

        if self.use_vgg16:
            vgg16 = models.vgg16(pretrained=True)
            self.conv = vgg16.features
            for param in self.conv.parameters():
                param.requires_grad = False
            self.fc_input_dim = 25088 + self.history_dim
            self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            dummy = torch.zeros(1, *self.input_dim)
            conv_out_size = self.conv(dummy).reshape(1, -1).size(1)
            self.fc_input_dim = conv_out_size + self.history_dim

        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, self.output_dim)

    def forward(self, state, history):
        batch_size = state.size(0)
        device = state.device

        # 1. Feature Extraction
        if self.use_vgg16:
            state = self.normalize(state)
            with torch.no_grad():
                features = self.conv(state).reshape(batch_size, -1)
        else:
            features = self.conv(state).reshape(batch_size, -1)
            
        x_fc_base = torch.cat([features, history], dim=1)

        pot_sum = torch.zeros(batch_size, self.output_dim, device=device)

        # 2. Spiking Temporal Loop
        mem1 = torch.zeros(batch_size, 128, device=device)
        mem2 = torch.zeros(batch_size, 256, device=device)
        mem3 = torch.zeros(batch_size, self.output_dim, device=device)
        
        syn1 = torch.zeros_like(mem1)
        syn2 = torch.zeros_like(mem2)
        syn3 = torch.zeros_like(mem3)

        for _ in range(self.simulation_time):
            # FC1
            h1 = self.fc1(x_fc_base)
            spk1, mem1, syn1 = self._spiking_neuron(h1, mem1, syn1)
            
            # FC2
            h2 = self.fc2(spk1)
            spk2, mem2, syn2 = self._spiking_neuron(h2, mem2, syn2)
            
            # FC3 (Accumulate potential, no spikes needed at output)
            h3 = self.fc3(spk2)
            syn3 = self.alpha * syn3 + h3
            mem3 = self.beta * mem3 + syn3
            
            pot_sum += mem3

        # Q-values are the average potentials
        return pot_sum / self.simulation_time

    def _spiking_neuron(self, h, mem, syn):
        new_syn = self.alpha * syn + h
        new_mem = self.beta * mem + new_syn
        
        mthr = new_mem - self.threshold
        out = self.spike_fn(mthr)
        
        # Soft reset
        res_mem = new_mem - (out * self.threshold)
        return out, res_mem, new_syn