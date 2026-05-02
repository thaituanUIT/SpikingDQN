"""Implementation for SNN with Surrogate Gradient learning"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from backbone.model import VGG16Backbone, SimpleConvBackbone, ResNetBackbone, FusionBackbone

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
                 simulation_time=10, alpha=0.9, beta=0.8, threshold=1.0, backbone_name='conv', dueling=False):
        super(SQNSurrogate, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.simulation_time = simulation_time
        self.backbone_name = backbone_name
        self.dueling = dueling

        
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.spike_fn = SuperSpike.apply

        if self.backbone_name == 'vgg16':
            self.backbone = VGG16Backbone()
        elif self.backbone_name == 'resnet18':
            self.backbone = ResNetBackbone(model_name='resnet18')
        elif self.backbone_name == 'fusion':
            self.backbone = FusionBackbone(model_name='resnet18')
        else:
            self.backbone = SimpleConvBackbone(input_channels=self.input_dim[0])
            
        self.fc_input_dim = self.backbone.get_output_dim() + self.history_dim

        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        
        if self.dueling:
            from backbone.engine import DuelingHead
            self.fc3 = DuelingHead(256, 128, self.output_dim)
        else:
            self.fc3 = nn.Linear(256, self.output_dim)

    def forward(self, state, history):
        batch_size = state.size(0)
        device = state.device

        # 1. Feature Extraction
        features = self.backbone(state)
            
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