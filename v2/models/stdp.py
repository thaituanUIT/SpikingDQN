"""Implementation for SNN with STDP learning"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoGFilter(nn.Module):
    def __init__(self, size=7, sigma1=1.0, sigma2=2.0):
        super(DoGFilter, self).__init__()
        # Simplified DoG filter
        x = torch.arange(size) - size // 2
        y = torch.arange(size) - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        g1 = torch.exp(-(xx**2 + yy**2) / (2 * sigma1**2)) / (2 * torch.pi * sigma1**2)
        g2 = torch.exp(-(xx**2 + yy**2) / (2 * sigma2**2)) / (2 * torch.pi * sigma2**2)
        
        dog = g1 - g2
        dog = dog - dog.mean() # Zero mean
        
        # Shape for Conv2d: (out_channels, in_channels, H, W)
        # We apply the same DoG over the 3 color channels 
        # (in practice, it's often run on grayscale, but we adapt it here)
        self.register_buffer('weight', dog.view(1, 1, size, size).repeat(3, 1, 1, 1))

    def forward(self, img_tensor):
        # Apply group convolution (each channel filtered independently)
        return F.conv2d(img_tensor, self.weight, padding=3, groups=3)

class STDPConv2d(nn.Module):
    """
    Simplified Convolutional Layer with Winner-Take-All and STDP.
    Operates on spike latencies (first-to-fire).
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, threshold=10.0):
        super(STDPConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.threshold = threshold
        
        # Weights initialized randomly [0.2, 0.8] and normalized
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size) * 0.6 + 0.2)
        self.normalize_weights()
        
        # STDP parameters
        self.lr_plus = 0.005 # Learning rate for depression/potentiation
        self.lr_minus = 0.0015
        
    def normalize_weights(self):
        """Keep sum of weights per neuron constant to prevent saturation"""
        with torch.no_grad():
            norm = self.weight.sum(dim=(1, 2, 3), keepdim=True)
            self.weight.data /= (norm + 1e-5)
            # Scaling factor to maintain reasonable potential levels
            self.weight.data *= (self.in_channels * self.kernel_size**2) * 0.5

    def forward(self, spike_latencies, is_training_stdp=False):
        """
        spike_latencies: (batch, in_channels, H, W) - Represents times of incoming spikes.
                         We assume lower value = earlier spike. 0 = no spike.
        """
        batch_size, _, H, W = spike_latencies.shape
        device = spike_latencies.device
        
        # Find continuous potentials over a simulated time window.
        # Since we use spatial convolutions in PyTorch, simulating literal time is slow.
        # Approximation: Convolve inverse latencies (stronger signal = earlier)
        
        # Transform latencies (smaller is better) to firing rates/potentials (larger is better).
        # We assume simulation max time is 15.
        T_max = 15.0
        # A spike is valid if its latency is > 0.
        active_mask = (spike_latencies > 0).float()
        potentials_in = (T_max - spike_latencies) * active_mask 
        
        # Conv
        potentials_out = F.conv2d(potentials_in, self.weight, padding=self.kernel_size//2)
        
        # Integrate and fire (Winner Take All)
        out_spikes = torch.zeros_like(potentials_out)
        
        # WTA over channels: only the feature map with the highest potential at (x,y) can fire
        max_potentials, winners = potentials_out.max(dim=1, keepdim=True)
        
        # Only those crossing threshold fire
        fire_mask = (max_potentials >= self.threshold).float()
        
        # Scatter spikes back to the winning channels
        out_spikes.scatter_(1, winners, fire_mask)
        
        if is_training_stdp:
            # Simplified STDP: Potentiate weights for the winning features based on input activity
            with torch.no_grad():
                # Unfold input to patches
                patches = F.unfold(active_mask, kernel_size=self.kernel_size, padding=self.kernel_size//2)
                patches = patches.view(batch_size, self.in_channels, self.kernel_size, self.kernel_size, H, W)
                
                # Flatten spatial and batch dimensions
                patches_permuted = patches.permute(0, 4, 5, 1, 2, 3).reshape(-1, self.in_channels, self.kernel_size, self.kernel_size)
                winners_flat = winners.view(-1)
                fire_mask_flat = fire_mask.view(-1)
                
                # Find where a spike actually occurred
                valid_indices = torch.nonzero(fire_mask_flat > 0).squeeze(-1)
                
                if valid_indices.numel() > 0:
                    valid_winners = winners_flat[valid_indices]
                    valid_pre_activity = patches_permuted[valid_indices]
                    
                    # Gather current weights for valid locations
                    w_selected = self.weight[valid_winners]
                    
                    # Calculate weight updates (LTP and LTD)
                    delta_w = self.lr_plus * valid_pre_activity * (1.0 - w_selected) \
                              - self.lr_minus * (1.0 - valid_pre_activity) * w_selected
                              
                    # Accumulate and apply updates
                    total_delta_w = torch.zeros_like(self.weight)
                    total_delta_w.index_add_(0, valid_winners, delta_w)
                    
                    self.weight += total_delta_w
                    self.weight.clamp_(0.01, 1.0) # Avoid absolute zero to allow recovery
                    self.normalize_weights()
                
        # Return out latencies (bounded mapping: higher potential -> earlier spike)
        # We map potential [threshold, inf) to latency (0, T_max]
        out_latencies = (T_max * self.threshold / (max_potentials + 1e-5)) * out_spikes
        return out_latencies

class SQNSTDP(nn.Module):
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90):
        super(SQNSTDP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        
        # Retinal Processing
        self.dog = DoGFilter()
        
        # Unsupervised STDP Backbone
        self.pool = nn.MaxPool2d(2, 2)
        # Increased thresholds significantly to ensure sparse, meaningful firing
        self.conv1 = STDPConv2d(3, 32, kernel_size=5, threshold=150.0)
        self.conv2 = STDPConv2d(32, 64, kernel_size=3, threshold=100.0)
        self.conv3 = STDPConv2d(64, 64, kernel_size=3, threshold=100.0)
        
        self.is_pretraining = False # Flag for STDP Phase vs RL Phase
        
        # Determine flattened conv feature size
        dummy = torch.ones(1, *self.input_dim)
        fc_input_dim = self.feature_size(dummy) + self.history_dim
        
        # RL Decision Head (Trained with backprop/DQN)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def set_pretrain_mode(self, mode):
        self.is_pretraining = mode
        if mode:
            # Freeze FC during STDP pretraining
            for param in self.fc.parameters():
                param.requires_grad = False
            for param in self.parameters():
                if isinstance(param, nn.Parameter) and "weight" in param.shape:
                    param.requires_grad = False # We update conv weights manually via STDP
        else:
            # Freeze Conv during RL
            for param in self.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = True

    def feature_size(self, x):
        with torch.no_grad():
            x = self.dog(x)
            # intensity to latency (simplification): higher intensity -> lower latency
            # Map intensity [0, 1] to latency [0.1, 15.0]
            latencies = (1.0 - x) * 14.9 + 0.1 
            
            x = self.conv1(latencies)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.pool(x)
            return x.reshape(1, -1).size(1)

    def forward(self, state, history):
        # 1. DoG Filtering
        with torch.no_grad():
            x = self.dog(state)
            
            # Simple Intensity-to-Latency Encoding (T_max = 15)
            # Normalize to [0, 1] then invert. Map to [0.1, 15.0] to ensure visibility to masks.
            x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min + 1e-5)
            latencies = (1.0 - x_norm) * 14.9 + 0.1
            
            # 2. STDP Convolutional Layers
            c1 = self.conv1(latencies, is_training_stdp=self.is_pretraining)
            c1 = self.pool(c1)
            
            c2 = self.conv2(c1, is_training_stdp=self.is_pretraining)
            c2 = self.pool(c2)
            
            c3 = self.conv3(c2, is_training_stdp=self.is_pretraining)
            c3 = self.pool(c3)
        
        # If in STDP pretraining phase, we don't care about RL output
        if self.is_pretraining:
            return torch.zeros(state.size(0), self.output_dim, device=state.device)
            
        # 3. RL Forward Pass
        features = c3.reshape(state.size(0), -1)
        x_fc = torch.cat([features, history], dim=1)
        
        q_values = self.fc(x_fc)
        return q_values