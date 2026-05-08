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
    Convolutional Layer with Winner-Take-All, STDP, and Adaptive Thresholds.
    Operates on spike latencies (first-to-fire).
    
    Key improvements over naive STDP:
    - Adaptive per-neuron thresholds with homeostasis (prevents dead/dominant neurons)
    - L2 weight normalization (preserves learned directional structure)
    - Scale-aware STDP updates (normalizes by fire count to stabilize LR)
    - Lateral inhibition for representational diversity
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, threshold=15.0,
                 target_firing_rate=0.05):
        super(STDPConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.base_threshold = threshold
        self.target_firing_rate = target_firing_rate
        
        # Weights initialized randomly [0.2, 0.8] and L2-normalized
        self.weight = nn.Parameter(
            torch.rand(out_channels, in_channels, kernel_size, kernel_size) * 0.6 + 0.2
        )
        self.normalize_weights()
        
        # Adaptive per-neuron thresholds (initialized to base)
        self.register_buffer(
            'adaptive_threshold', 
            torch.full((out_channels,), threshold)
        )
        # Exponential moving average of firing rates per neuron
        self.register_buffer(
            'firing_rate_ema',
            torch.full((out_channels,), target_firing_rate)
        )
        
        # STDP parameters
        self.lr_plus = 0.004   # LTP (potentiation)
        self.lr_minus = 0.0012 # LTD (depression)
        
        # Homeostasis parameters
        self.homeostasis_rate = 0.01  # How fast thresholds adapt
        self.ema_decay = 0.99         # EMA smoothing for firing rate tracking
        
    def normalize_weights(self):
        """L2-normalize weights per neuron to preserve directional structure"""
        with torch.no_grad():
            # L2 norm per output neuron (across in_channels, kH, kW)
            norm = torch.norm(
                self.weight.data.view(self.weight.size(0), -1), 
                dim=1, keepdim=True
            ).view(-1, 1, 1, 1)
            self.weight.data /= (norm + 1e-6)
            # Scale to a reasonable magnitude for potential computation
            fan_in = self.in_channels * self.kernel_size ** 2
            self.weight.data *= (fan_in ** 0.5)

    def forward(self, spike_latencies, is_training_stdp=False):
        """
        spike_latencies: (batch, in_channels, H, W) - Represents times of incoming spikes.
                         Lower value = earlier spike. 0 = no spike.
        """
        batch_size, _, H, W = spike_latencies.shape
        device = spike_latencies.device
        
        T_max = 15.0
        # A spike is valid if its latency is > 0.
        active_mask = (spike_latencies > 0).float()
        potentials_in = (T_max - spike_latencies) * active_mask 
        
        # Conv to compute output potentials
        potentials_out = F.conv2d(potentials_in, self.weight, padding=self.kernel_size//2)
        
        # --- Adaptive Threshold Integration ---
        # Shape thresholds for broadcasting: (1, out_channels, 1, 1)
        thresholds = self.adaptive_threshold.view(1, -1, 1, 1)
        
        # Integrate and fire (Winner Take All)
        out_spikes = torch.zeros_like(potentials_out)
        
        # WTA over channels: only the feature map with the highest potential at (x,y) can fire
        max_potentials, winners = potentials_out.max(dim=1, keepdim=True)
        
        # Gather the threshold for the winning neuron at each spatial position
        winner_thresholds = thresholds.expand_as(potentials_out).gather(1, winners)
        
        # Only those crossing their adaptive threshold fire
        fire_mask = (max_potentials >= winner_thresholds).float()
        
        # Scatter spikes back to the winning channels
        out_spikes.scatter_(1, winners, fire_mask)
        
        # --- Homeostasis: Update adaptive thresholds ---
        if is_training_stdp:
            with torch.no_grad():
                # Compute per-neuron firing rate for this batch
                # out_spikes shape: (batch, out_channels, H, W)
                spatial_count = H * W * batch_size
                current_firing_rate = out_spikes.sum(dim=(0, 2, 3)) / (spatial_count + 1e-8)
                
                # Update EMA of firing rates
                self.firing_rate_ema = (
                    self.ema_decay * self.firing_rate_ema + 
                    (1 - self.ema_decay) * current_firing_rate
                )
                
                # Adjust thresholds: increase if firing too much, decrease if too little
                rate_error = self.firing_rate_ema - self.target_firing_rate
                self.adaptive_threshold += self.homeostasis_rate * rate_error * self.adaptive_threshold
                
                # Clamp thresholds to reasonable range
                self.adaptive_threshold.clamp_(
                    self.base_threshold * 0.1,  # Don't drop below 10% of base
                    self.base_threshold * 5.0   # Don't exceed 5x base
                )
        
        # --- STDP Weight Updates ---
        if is_training_stdp:
            with torch.no_grad():
                # Unfold input to patches
                patches = F.unfold(active_mask, kernel_size=self.kernel_size, 
                                   padding=self.kernel_size//2)
                patches = patches.view(
                    batch_size, self.in_channels, self.kernel_size, self.kernel_size, H, W
                )
                
                # Flatten spatial and batch dimensions
                patches_permuted = patches.permute(0, 4, 5, 1, 2, 3).reshape(
                    -1, self.in_channels, self.kernel_size, self.kernel_size
                )
                winners_flat = winners.view(-1)
                fire_mask_flat = fire_mask.view(-1)
                
                # Find where a spike actually occurred
                valid_indices = torch.nonzero(fire_mask_flat > 0).squeeze(-1)
                
                if valid_indices.numel() > 0:
                    valid_winners = winners_flat[valid_indices]
                    valid_pre_activity = patches_permuted[valid_indices]
                    
                    # Gather current weights for valid locations
                    w_selected = self.weight[valid_winners]
                    
                    # Normalize weights to [0,1] for STDP rule computation
                    w_min = w_selected.min()
                    w_max = w_selected.max()
                    w_range = w_max - w_min + 1e-6
                    w_normed = (w_selected - w_min) / w_range
                    
                    # Calculate weight updates (LTP and LTD)
                    delta_w = (self.lr_plus * valid_pre_activity * (1.0 - w_normed) 
                               - self.lr_minus * (1.0 - valid_pre_activity) * w_normed)
                    
                    # --- Scale-aware: average per neuron instead of sum ---
                    # Count how many times each neuron won
                    neuron_fire_counts = torch.zeros(
                        self.out_channels, device=device
                    )
                    neuron_fire_counts.scatter_add_(
                        0, valid_winners, torch.ones_like(valid_winners, dtype=torch.float)
                    )
                    
                    # Accumulate deltas
                    total_delta_w = torch.zeros_like(self.weight)
                    total_delta_w.index_add_(0, valid_winners, delta_w)
                    
                    # Normalize by fire count (prevent LR scaling with spatial resolution)
                    for n in range(self.out_channels):
                        if neuron_fire_counts[n] > 0:
                            total_delta_w[n] /= neuron_fire_counts[n]
                    
                    self.weight.data += total_delta_w
                    self.weight.data.clamp_(0.01, None)  # Floor only, no ceiling
                    self.normalize_weights()
                
        # Return output latencies (higher potential -> earlier spike)
        out_latencies = (T_max * self.base_threshold / (max_potentials + 1e-5)) * out_spikes
        return out_latencies


class SQNSTDP(nn.Module):
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90, dueling=False):
        super(SQNSTDP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.dueling = dueling
        
        # Retinal Processing
        self.dog = DoGFilter()
        
        # Unsupervised STDP Backbone
        self.pool = nn.MaxPool2d(2, 2)
        # Lowered thresholds to match realistic potential magnitudes
        self.conv1 = STDPConv2d(3, 32, kernel_size=5, threshold=15.0, target_firing_rate=0.05)
        self.conv2 = STDPConv2d(32, 64, kernel_size=3, threshold=12.0, target_firing_rate=0.05)
        self.conv3 = STDPConv2d(64, 64, kernel_size=3, threshold=12.0, target_firing_rate=0.05)
        
        self.is_pretraining = False # Flag for STDP Phase vs RL Phase
        
        # Determine flattened conv feature size
        dummy = torch.ones(1, *self.input_dim)
        conv_out_dim = self._compute_conv_output_dim(dummy)
        fc_input_dim = conv_out_dim + self.history_dim
        
        # Feature normalization layer (learnable, trains with RL head)
        self.feature_norm = nn.LayerNorm(conv_out_dim)
        
        # RL Decision Head (Trained with backprop/DQN)
        if self.dueling:
            from backbone.engine import DuelingHead
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                DuelingHead(256, 128, self.output_dim)
            )
        else:
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
            # Freeze FC and feature_norm during STDP pretraining
            for param in self.fc.parameters():
                param.requires_grad = False
            for param in self.feature_norm.parameters():
                param.requires_grad = False
        else:
            # Freeze Conv during RL, unfreeze FC + feature_norm
            for param in self.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = True
            for param in self.feature_norm.parameters():
                param.requires_grad = True

    def _encode_latencies(self, dog_output):
        """
        Convert DoG filter output to spike latencies.
        Only positive DoG responses (detected features) produce spikes.
        Negative responses are suppressed (latency = 0 = no spike).
        """
        T_max = 15.0
        
        # Separate positive responses (feature detections)
        positive_mask = (dog_output > 0).float()
        
        # Per-channel normalization of positive values to [0, 1]
        # Flatten spatial dims for per-channel min/max
        pos_values = dog_output * positive_mask
        
        # Per-sample, per-channel max (for proper normalization)
        ch_max = pos_values.amax(dim=(-2, -1), keepdim=True)
        ch_max = ch_max.clamp(min=1e-6)  # Avoid division by zero
        
        # Normalize positive values to [0, 1]
        normalized = pos_values / ch_max
        
        # Map to latencies: stronger feature -> lower latency (earlier spike)
        # Range: [0.1, T_max] for active spikes, 0 for no spike
        latencies = ((1.0 - normalized) * (T_max - 0.1) + 0.1) * positive_mask
        
        return latencies

    def _compute_conv_output_dim(self, x):
        """Compute flattened dimension of conv backbone output"""
        with torch.no_grad():
            x = self.dog(x)
            latencies = self._encode_latencies(x)
            
            x = self.conv1(latencies)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.pool(x)
            return x.reshape(1, -1).size(1)

    # Keep old name for backward compat with train.py feature_size calls
    def feature_size(self, x):
        return self._compute_conv_output_dim(x)

    def forward(self, state, history):
        # 1. DoG Filtering
        with torch.no_grad():
            x = self.dog(state)
            
            # 2. Proper intensity-to-latency encoding
            latencies = self._encode_latencies(x)
            
            # 3. STDP Convolutional Layers
            c1 = self.conv1(latencies, is_training_stdp=self.is_pretraining)
            c1 = self.pool(c1)
            
            c2 = self.conv2(c1, is_training_stdp=self.is_pretraining)
            c2 = self.pool(c2)
            
            c3 = self.conv3(c2, is_training_stdp=self.is_pretraining)
            c3 = self.pool(c3)
        
        # If in STDP pretraining phase, we don't care about RL output
        if self.is_pretraining:
            return torch.zeros(state.size(0), self.output_dim, device=state.device)
            
        # 4. RL Forward Pass with feature normalization
        features = c3.reshape(state.size(0), -1)
        
        # Normalize features so FC head sees a stable input distribution
        features = self.feature_norm(features)
        
        x_fc = torch.cat([features, history], dim=1)
        
        q_values = self.fc(x_fc)
        return q_values
    
    def get_backbone_stats(self):
        """Return diagnostic statistics for the STDP backbone layers"""
        stats = {}
        for name, layer in [('conv1', self.conv1), ('conv2', self.conv2), ('conv3', self.conv3)]:
            stats[name] = {
                'threshold_mean': layer.adaptive_threshold.mean().item(),
                'threshold_std': layer.adaptive_threshold.std().item(),
                'firing_rate_mean': layer.firing_rate_ema.mean().item(),
                'firing_rate_std': layer.firing_rate_ema.std().item(),
                'weight_mean': layer.weight.data.mean().item(),
                'weight_std': layer.weight.data.std().item(),
            }
        return stats