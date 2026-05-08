import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer

# Import from v3 to ensure we use the correct version of the backbone
from backbone.model import VGG16Backbone, SimpleConvBackbone, ResNetBackbone, FusionBackbone

class SQNJelly(nn.Module):
    """
    Implementation of Spiking Q-Network using SpikingJelly components.
    This model uses MultiStep modules for efficient parallel temporal simulation.
    """
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90, 
                 simulation_time=10, tau=2.0, threshold=1.0, backbone_name='conv'):
        super(SQNJelly, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.simulation_time = simulation_time
        self.backbone_name = backbone_name
        
        # 1. Feature Extractor (Backbone)
        if self.backbone_name == 'vgg16':
            self.backbone = VGG16Backbone()
        elif self.backbone_name == 'resnet18':
            self.backbone = ResNetBackbone(model_name='resnet18')
        elif self.backbone_name == 'fusion':
            self.backbone = FusionBackbone(model_name='resnet18')
        else:
            self.backbone = SimpleConvBackbone(input_channels=self.input_dim[0])
            
        self.fc_input_dim = self.backbone.get_output_dim() + self.history_dim

        # 2. Spiking Layers
        # We use LIFNode with step_mode='m' to process the entire simulation time at once
        self.fc1 = layer.MultiStepContainer(nn.Linear(self.fc_input_dim, 128))
        self.lif1 = neuron.LIFNode(tau=tau, v_threshold=threshold, 
                                   surrogate_function=surrogate.ATan(), detach_reset=True,
                                   step_mode='m')
        
        self.fc2 = layer.MultiStepContainer(nn.Linear(128, 256))
        self.lif2 = neuron.LIFNode(tau=tau, v_threshold=threshold, 
                                   surrogate_function=surrogate.ATan(), detach_reset=True,
                                   step_mode='m')
        
        self.fc3 = layer.MultiStepContainer(nn.Linear(256, self.output_dim))
        # The output layer integrates potentials to provide Q-values.
        # We use a LIFNode with infinite threshold to act as a pure leaky integrator.
        self.lif3 = neuron.LIFNode(tau=tau, v_threshold=float('inf'), 
                                   detach_reset=True, step_mode='m')

    def forward(self, state, history):
        batch_size = state.size(0)
        device = state.device
        
        # 1. Static Feature Extraction
        features = self.backbone(state)
        x_fc_base = torch.cat([features, history], dim=1) # (batch, fc_input_dim)
        
        # 2. Expand for Temporal Dimension (T, Batch, Features)
        x_seq = x_fc_base.unsqueeze(0).repeat(self.simulation_time, 1, 1)
        
        # 3. Multi-Step Forward Pass
        x_seq = self.fc1(x_seq)
        x_seq = self.lif1(x_seq)
        
        x_seq = self.fc2(x_seq)
        x_seq = self.lif2(x_seq)
        
        x_seq = self.fc3(x_seq)
        
        # 4. For the final layer, we want the average membrane potential.
        # In MultiStep mode, LIFNode returns spikes (which will be 0 due to inf threshold).
        # We can use a custom loop or a monitor to get 'v'.
        # For simplicity and to match v2's logic, we'll manually integrate the last layer
        # since it's just a small linear output.
        
        q_values_sum = torch.zeros(batch_size, self.output_dim, device=device)
        v = torch.zeros(batch_size, self.output_dim, device=device)
        
        tau_val = self.lif3.tau
        for t in range(self.simulation_time):
            # Leaky integration: v[t] = v[t-1] * (1 - 1/tau) + x[t] / tau
            # (Matches SpikingJelly's internal LIF logic)
            v = v + (x_seq[t] - v) / tau_val
            q_values_sum += v
            
        # 5. Q-Values: Average membrane potential over simulation time
        q_values = q_values_sum / self.simulation_time
        
        # 6. Reset network state
        functional.reset_net(self)
        
        return q_values
