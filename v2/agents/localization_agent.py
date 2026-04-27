import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from data.preprocess import crop_and_resize

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.buffer)

class LocalizationAgent:
    def __init__(self, model, optimizer=None, device='cpu', 
                 gamma=0.1, max_steps=20, action_options=9, history_size=10,
                 clip_grad=1.0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        self.gamma = gamma
        self.max_steps = max_steps
        self.action_options = action_options
        self.history_size = history_size
        
        self.memory = ReplayBuffer(capacity=1000)
        self.loss_fn = nn.HuberLoss()
        self.clip_grad = clip_grad
        
    def get_action(self, image_tensor, history_tensor, epsilon, current_mask, ground_truth):
        """
        Selects action using epsilon-greedy policy. 
        If training (ground_truth provided), it can fallback to positive reward actions.
        """
        if random.random() > epsilon:
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(image_tensor.to(self.device), history_tensor.to(self.device))
            self.model.train()
            action = torch.argmax(q_values).item()
        else:
            # Random action exploration guided by positive reward (like v1)
            finish_reward = self.compute_finish_reward(current_mask, ground_truth)
            if finish_reward > 0:
                action = 8
            else:
                rewards = []
                for i in range(self.action_options - 1):
                    reward = self.compute_reward(i, current_mask, ground_truth)
                    rewards.append(reward)
                positive_idx = np.where(np.array(rewards) >= 0)[0]
                
                if len(positive_idx) == 0:
                    action = random.choice(range(8))
                else:
                    action = random.choice(positive_idx)
        return action

    def compute_mask(self, action, current_mask):
        image_rate = 0.1
        delta_width = image_rate * (current_mask[2] - current_mask[0])
        delta_height = image_rate * (current_mask[3] - current_mask[1])
        dx1, dy1, dx2, dy2 = 0, 0, 0, 0

        if action == 0:
            dx1 = delta_width; dx2 = delta_width
        elif action == 1:
            dx1 = -delta_width; dx2 = -delta_width
        elif action == 2:
            dy1 = delta_height; dy2 = delta_height
        elif action == 3:
            dy1 = -delta_height; dy2 = -delta_height
        elif action == 4:
            dx1 = -delta_width; dx2 = delta_width
            dy1 = -delta_height; dy2 = delta_height
        elif action == 5:
            dx1 = delta_width; dx2 = -delta_width
            dy1 = delta_height; dy2 = -delta_height
        elif action == 6:
            dy1 = delta_height; dy2 = -delta_height
        elif action == 7:
            dx1 = delta_width; dx2 = -delta_width

        new_mask_tmp = np.array([current_mask[0] + dx1, current_mask[1] + dy1,
                                 current_mask[2] + dx2, current_mask[3] + dy2])
        new_mask = np.array([
            min(new_mask_tmp[0], new_mask_tmp[2]),
            min(new_mask_tmp[1], new_mask_tmp[3]),
            max(new_mask_tmp[0], new_mask_tmp[2]),
            max(new_mask_tmp[1], new_mask_tmp[3])
        ])
        return new_mask

    def compute_iou(self, mask, ground_truth):
        dx = min(mask[2], ground_truth[2]) - max(mask[0], ground_truth[0])
        dy = min(mask[3], ground_truth[3]) - max(mask[1], ground_truth[1])

        inter_area = dx * dy if (dx >= 0) and (dy >= 0) else 0

        mask_area = (mask[2] - mask[0]) * (mask[3] - mask[1])
        ground_truth_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])

        union = mask_area + ground_truth_area - inter_area
        return inter_area / union if union > 0 else 0

    def compute_reward(self, action, current_mask, ground_truth):
        new_mask = self.compute_mask(action, current_mask)
        iou_new = self.compute_iou(new_mask, ground_truth)
        iou_current = self.compute_iou(current_mask, ground_truth)

        # Basic reward structure: positive if IoU improves
        return 1 if iou_new > iou_current else -1

    def compute_finish_reward(self, current_mask, ground_truth):
        # +3 if terminated with IoU > 0.5, else -3
        return 3 if self.compute_iou(current_mask, ground_truth) > 0.5 else -3

    def feature_extract(self, img, history, width, height, current_mask):
        """Converts mask to cropped image tensor and history list to tensor"""
        cropped_img = crop_and_resize(img, current_mask)
        img_transposed = np.transpose(cropped_img, (2, 0, 1)) 
        
        # SNN models might expect batches, so we add batch dim
        image_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float() / 255.0
        
        feat_hist = np.zeros(self.action_options * self.history_size)
        for i, act in enumerate(history):
            if act != -1:
                feat_hist[i * self.action_options + act] = 1
        history_tensor = torch.tensor(feat_hist).float().unsqueeze(0)
        
        return image_tensor, history_tensor

    def step(self, image, history, current_mask, ground_truth, step_count, epsilon):
        height, width, _ = image.shape
        image_tensor, history_tensor = self.feature_extract(image, history, width, height, current_mask)
        
        # Handle maximum steps termination
        if step_count >= self.max_steps:
            action = 8
        else:
            action = self.get_action(image_tensor, history_tensor, epsilon, current_mask, ground_truth)

        if action == 8:
            new_mask = current_mask
            reward = self.compute_finish_reward(current_mask, ground_truth)
            done = True
        else:
            new_mask = self.compute_mask(action, current_mask)
            reward = self.compute_reward(action, current_mask, ground_truth)
            history = history[1:] + [action]
            done = False

        next_image_tensor, next_history_tensor = self.feature_extract(image, history, width, height, new_mask)
        
        # Store transition in numpy for simplicity in replay buffer
        state = {'image': image_tensor.numpy()[0], 'history': history_tensor.numpy()[0]}
        next_state = {'image': next_image_tensor.numpy()[0], 'history': next_history_tensor.numpy()[0]}
        
        self.memory.push(state, action, next_state, reward, done)
        
        return new_mask, reward, done, history

    def train_step(self, batch_size=20):
        if len(self.memory) < batch_size or not self.optimizer:
            return 0.0

        states, actions, next_states, rewards, dones = self.memory.sample(batch_size)
        
        img_states = torch.FloatTensor(np.stack([s['image'] for s in states])).to(self.device)
        hist_states = torch.FloatTensor(np.stack([s['history'] for s in states])).to(self.device)
        
        img_next = torch.FloatTensor(np.stack([s['image'] for s in next_states])).to(self.device)
        hist_next = torch.FloatTensor(np.stack([s['history'] for s in next_states])).to(self.device)
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        self.optimizer.zero_grad()
        
        # Current Q-Values
        q_vals = self.model(img_states, hist_states)
        q_acts = q_vals.gather(1, torch.LongTensor(actions).to(self.device).unsqueeze(-1)).squeeze(-1)
        
        # Next Q-Values
        self.model.eval()
        with torch.no_grad():
            next_q_vals = self.model(img_next, hist_next)
            max_next_q = next_q_vals.max(dim=1)[0]
        self.model.train()
        
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        loss = self.loss_fn(q_acts, target_q)
        loss.backward()
        
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
        self.optimizer.step()
        
        return loss.item()
