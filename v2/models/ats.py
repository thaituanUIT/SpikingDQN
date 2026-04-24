"""Implementation for SNN converted from ANN (ANN-to-SNN)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

class SQNConverted(nn.Module):
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90, simulation_time=10, use_vgg16=False):
        super(SQNConverted, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.simulation_time = simulation_time
        self.use_vgg16 = use_vgg16
        
        self.is_snn = False # Flag indicating if it has been converted
        
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

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state, history):
        if not self.is_snn:
            # Standard ANN Forward pass
            if self.use_vgg16:
                state = self.normalize(state)
                with torch.no_grad():
                    features = self.conv(state).reshape(state.size(0), -1)
            else:
                features = self.conv(state).reshape(state.size(0), -1)
                
            x = torch.cat([features, history], dim=1)
            q_values = self.fc(x)
            return q_values
        else:
            # SNN Forward pass (Integrate and Fire simulation)
            state_size = state.size(0)
            device = state.device
            
            out_v = torch.zeros(state_size, self.output_dim, device=device)
            
            # Since VGG16 relies on heavy pooling and continuous weights, 
            # ATS conversion normally skips VGG and only applies to the trained RL head
            # Or we can treat VGG output as a constant current.
            if self.use_vgg16:
                state = self.normalize(state)
                with torch.no_grad():
                    constant_features = self.conv(state).reshape(state_size, -1)
            
            mem_conv = [None] * 4 # Adjust if needed
            mem_fc = [None] * 2
            
            # We assume input is constant current over time
            for t in range(self.simulation_time):
                x_in = state
                
                if self.use_vgg16:
                    features = constant_features
                else:
                    # Manual pass through layers to track membrane potentials
                    c_idx = 0
                    for layer in self.conv:
                        if isinstance(layer, nn.Conv2d):
                            x_in = layer(x_in)
                        elif isinstance(layer, nn.ReLU):
                            if mem_conv[c_idx] is None:
                                mem_conv[c_idx] = torch.zeros_like(x_in)
                            mem_conv[c_idx] += x_in
                            spikes = (mem_conv[c_idx] >= 1.0).float()
                            mem_conv[c_idx] -= spikes
                            x_in = spikes
                            c_idx += 1
                            
                    features = x_in.reshape(state_size, -1)
                    
                x_in = torch.cat([features, history], dim=1)
                
                f_idx = 0
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                        x_in = layer(x_in)
                    elif isinstance(layer, nn.ReLU):
                        if mem_fc[f_idx] is None:
                            mem_fc[f_idx] = torch.zeros_like(x_in)
                        mem_fc[f_idx] += x_in
                        spikes = (mem_fc[f_idx] >= 1.0).float()
                        mem_fc[f_idx] -= spikes
                        x_in = spikes
                        f_idx += 1
                
                out_v += x_in # Last layer is Linear (no ReLU), acts as voltage accumulator

            return out_v / self.simulation_time

    def convert_to_snn(self, dataloader=None):
        """
        Locks the network and converts it to an SNN.
        For rigorous ATS, one would perform Data-Based Normalization here 
        by finding max activations on the dataloader and rescaling weights. 
        We just set the flag for this simplified version.
        """
        self.is_snn = True
        self.eval()
        print("Model converted to SNN mode.")