"""Implementation for SNN converted from ANN (ANN-to-SNN)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from backbone.model import (
    VGG16Backbone, SimpleConvBackbone, ResNetBackbone, FusionBackbone,
    ViTBackbone, EfficientNetBackbone, MobileNetBackbone
)

class SQNConverted(nn.Module):
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90, simulation_time=10, backbone_name='conv', dueling=False):
        super(SQNConverted, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.simulation_time = simulation_time
        self.backbone_name = backbone_name
        self.dueling = dueling
        
        self.is_snn = False # Flag indicating if it has been converted
        
        # 1. Khởi tạo Backbone
        if self.backbone_name == 'vgg16':
            self.backbone = VGG16Backbone()
        elif self.backbone_name == 'resnet18':
            self.backbone = ResNetBackbone(model_name='resnet18')
        elif self.backbone_name == 'fusion':
            self.backbone = FusionBackbone(model_name='resnet18')
        elif self.backbone_name == 'vit':
            self.backbone = ViTBackbone(model_name='vit_b_16')
        elif self.backbone_name == 'efficientnet':
            self.backbone = EfficientNetBackbone(model_name='efficientnet_b0')
        elif self.backbone_name == 'mobilenet':
            self.backbone = MobileNetBackbone(model_name='mobilenet_v3_small')
        else:
            self.backbone = SimpleConvBackbone(input_channels=self.input_dim[0])
            
        self.fc_input_dim = self.backbone.get_output_dim() + self.history_dim

        # 2. Xác định Final Layer trước
        if self.dueling:
            from backbone.engine import DuelingHead
            final_layer = DuelingHead(64, 32, self.output_dim)
        else:
            final_layer = nn.Linear(64, self.output_dim)

        # 3. Khởi tạo FC Layers một lần duy nhất
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            
            final_layer
        )

    def forward(self, state, history):
        if not self.is_snn:
            # Standard ANN Forward pass
            features = self.backbone(state)
                
            x = torch.cat([features, history], dim=1)
            q_values = self.fc(x)
            return q_values
        else:
            
            # SNN Forward pass (Integrate and Fire simulation)
            state_size = state.size(0)
            device = state.device
            
            out_v = torch.zeros(state_size, self.output_dim, device=device)
            
            # ATS conversion normally skips Backbone and applies to the trained RL head
            if self.backbone_name in ['vgg16', 'resnet18', 'fusion', 'vit', 'efficientnet', 'mobilenet']:
                with torch.no_grad():
                    constant_features = self.backbone(state)
            
            mem_conv = {}
            mem_fc = {}
            
            # We assume input is constant current over time
            for t in range(self.simulation_time):
                x_in = state
                
                if self.backbone_name in ['vgg16', 'resnet18', 'fusion', 'vit', 'efficientnet', 'mobilenet']:
                    features = constant_features
                else:
                    # Manual pass through layers to track membrane potentials
                    c_idx = 0
                    for layer in self.backbone.get_layers():
                        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Flatten, nn.Linear)):
                            x_in = layer(x_in)
                        elif isinstance(layer, nn.ReLU):
                            if c_idx not in mem_conv:
                                mem_conv[c_idx] = torch.zeros_like(x_in)
                            mem_conv[c_idx] += x_in
                            spikes = (mem_conv[c_idx] >= 1.0).float()
                            mem_conv[c_idx] -= spikes
                            x_in = spikes
                            c_idx += 1
                            
                    features = x_in.reshape(state_size, -1)
                    
                x_in = torch.cat([features, history], dim=1)
                
                if self.dueling:
                    from backbone.engine import DuelingHead
                    valid_layers = (nn.Linear, DuelingHead)
                else:
                    valid_layers = (nn.Linear,)

                f_idx = 0
                for layer in self.fc:
                    if isinstance(layer, valid_layers):
                        x_in = layer(x_in)
                    elif isinstance(layer, nn.ReLU):
                        if f_idx not in mem_fc:
                            mem_fc[f_idx] = torch.zeros_like(x_in)
                        mem_fc[f_idx] += x_in
                        spikes = (mem_fc[f_idx] >= 1.0).float()
                        mem_fc[f_idx] -= spikes
                        x_in = spikes
                        f_idx += 1
                    # Lưu ý: Các lớp nn.Dropout trong SNN sẽ tự động bị bỏ qua (bypass) 
                    # vì trong vòng lặp này chúng ta không xử lý instance của nn.Dropout. 
                    # Điều này là CHÍNH XÁC vì khi inference SNN (is_snn=True), hàm eval() 
                    # cũng khiến Dropout không có tác dụng.
                
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
