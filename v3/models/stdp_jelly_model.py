import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, layer, learning, encoding

class DoGFilter(nn.Module):
    """Retinal Processing: Difference of Gaussians Filter (Ported from v2)"""
    def __init__(self, size=7, sigma1=1.0, sigma2=2.0):
        super(DoGFilter, self).__init__()
        x = torch.arange(size) - size // 2
        y = torch.arange(size) - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        g1 = torch.exp(-(xx**2 + yy**2) / (2 * sigma1**2)) / (2 * torch.pi * sigma1**2)
        g2 = torch.exp(-(xx**2 + yy**2) / (2 * sigma2**2)) / (2 * torch.pi * sigma2**2)
        
        dog = g1 - g2
        dog = dog - dog.mean()
        
        self.register_buffer('weight', dog.view(1, 1, size, size).repeat(3, 1, 1, 1))

    def forward(self, img_tensor):
        return F.conv2d(img_tensor, self.weight, padding=3, groups=3)

class SQNSTDPJelly(nn.Module):
    """
    Spiking Q-Network with STDP-trained backbone using SpikingJelly.
    The backbone is trained unsupervised via STDP, and the head is trained via RL (Backprop).
    """
    def __init__(self, input_dim=(3, 224, 224), output_dim=9, history_dim=90, 
                 simulation_time=20):
        super(SQNSTDPJelly, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.history_dim = history_dim
        self.simulation_time = simulation_time
        self.is_pretraining = False # Phase flag: True for STDP, False for RL
        
        # 1. Retinal Filter
        self.dog = DoGFilter()
        
        # 2. SNN Backbone (STDP trained)
        # We use single-step mode for the backbone to apply STDP updates at each step
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
        self.lif1 = neuron.IFNode(v_threshold=1.0)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.lif2 = neuron.IFNode(v_threshold=1.0)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.lif3 = neuron.IFNode(v_threshold=1.0)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 3. STDP Learners
        # stdp_params: learning rates for potentiation and depression
        stdp_lr = (0.01, -0.005) 
        self.learner1 = learning.STDPLearner(step_mode='s', synapse=self.conv1, sn=self.lif1, 
                                             tau_pre=20.0, tau_post=20.0)
        self.learner2 = learning.STDPLearner(step_mode='s', synapse=self.conv2, sn=self.lif2, 
                                             tau_pre=20.0, tau_post=20.0)
        self.learner3 = learning.STDPLearner(step_mode='s', synapse=self.conv3, sn=self.lif3, 
                                             tau_pre=20.0, tau_post=20.0)
        
        # Encoder to convert intensity to spikes
        self.encoder = encoding.PoissonEncoder()

        # 4. Determine flattened feature size
        dummy_state = torch.zeros(1, *input_dim)
        dummy_history = torch.zeros(1, history_dim)
        with torch.no_grad():
            features = self.get_backbone_features(dummy_state)
            self.fc_input_dim = features.shape[1] + history_dim

        # 5. RL Head (ANN Decision layer)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def set_pretrain_mode(self, mode):
        self.is_pretraining = mode
        # Freeze/Unfreeze based on phase
        if mode:
            # Unsupervised phase: only STDP backbone learns (manually via learners)
            self.fc.requires_grad_(False)
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.conv3.weight.requires_grad = False
        else:
            # RL phase: Backprop on RL head, freeze backbone
            self.fc.requires_grad_(True)
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.conv3.weight.requires_grad = False

    def get_backbone_features(self, state):
        batch_size = state.size(0)
        device = state.device
        
        # Initial DoG filtering
        x = self.dog(state)
        
        # Normalization for Poisson encoding
        x = (x - x.min()) / (x.max() - x.min() + 1e-5)
        
        # Simulation Loop
        for t in range(self.simulation_time):
            # Encode to spikes
            spk_in = self.encoder(x)
            
            # Layer 1
            out1 = self.conv1(spk_in)
            spk1 = self.lif1(out1)
            if self.is_pretraining:
                self.learner1.step(on_grad=False)
            spk1 = self.pool(spk1)
            
            # Layer 2
            out2 = self.conv2(spk1)
            spk2 = self.lif2(out2)
            if self.is_pretraining:
                self.learner2.step(on_grad=False)
            spk2 = self.pool(spk2)
            
            # Layer 3
            out3 = self.conv3(spk2)
            spk3 = self.lif3(out3)
            if self.is_pretraining:
                self.learner3.step(on_grad=False)
            spk3 = self.pool(spk3)
            
        # Extract features from the final layer of the last timestep
        features = spk3.reshape(batch_size, -1)
        
        # Reset after each full simulation
        functional.reset_net(self)
        
        return features

    def forward(self, state, history):
        # 1. Get features from STDP backbone
        features = self.get_backbone_features(state)
        
        if self.is_pretraining:
            # During pre-training, we don't use the RL head
            return torch.zeros(state.size(0), self.output_dim, device=state.device)
            
        # 2. RL Decision
        x_fc = torch.cat([features, history], dim=1)
        q_values = self.fc(x_fc)
        return q_values
