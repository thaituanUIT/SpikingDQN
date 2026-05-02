import torch
import torch.nn as nn
import copy

class DQNEngine:
    """Standard Deep Q-Network Engine with an optional Target Network."""
    def __init__(self, model, gamma=0.9, use_target_net=True):
        self.model = model
        self.gamma = gamma
        self.use_target_net = use_target_net
        
        if self.use_target_net:
            self.target_model = copy.deepcopy(model)
            self.target_model.eval()
        else:
            self.target_model = self.model
            
    def update_target(self):
        """Updates the target network weights with the current model weights."""
        if self.use_target_net:
            self.target_model.load_state_dict(self.model.state_dict())

    def compute_loss(self, states_img, states_hist, actions, rewards, next_states_img, next_states_hist, dones, loss_fn, device):
        # Current Q-Values
        q_vals = self.model(states_img, states_hist)
        q_acts = q_vals.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Next Q-Values using Target Network
        self.target_model.eval()
        with torch.no_grad():
            next_q_vals = self.target_model(next_states_img, next_states_hist)
            max_next_q = next_q_vals.max(dim=1)[0]
            
        # Target Q calculation
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        loss = loss_fn(q_acts, target_q)
        return loss

class DoubleDQNEngine(DQNEngine):
    """Double DQN Engine to alleviate overestimation bias."""
    def compute_loss(self, states_img, states_hist, actions, rewards, next_states_img, next_states_hist, dones, loss_fn, device):
        # Current Q-Values
        q_vals = self.model(states_img, states_hist)
        q_acts = q_vals.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Action selection from Current Network
        self.model.eval()
        with torch.no_grad():
            next_q_vals_eval = self.model(next_states_img, next_states_hist)
            best_actions = next_q_vals_eval.max(dim=1)[1]
        self.model.train()
        
        # Q-Value evaluation from Target Network
        self.target_model.eval()
        with torch.no_grad():
            next_q_vals_target = self.target_model(next_states_img, next_states_hist)
            max_next_q = next_q_vals_target.gather(1, best_actions.unsqueeze(-1)).squeeze(-1)
            
        # Target Q calculation
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        loss = loss_fn(q_acts, target_q)
        return loss

class DuelingHead(nn.Module):
    """
    Standard ANN Dueling Head to replace final classification layers.
    Splits features into a Value stream and an Advantage stream.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DuelingHead, self).__init__()
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
