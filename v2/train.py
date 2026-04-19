import argparse
import torch
import torch.optim as optim
import os
import numpy as np

from v2.data.voc import VOCDataset
from v2.agents.localization_agent import LocalizationAgent
from v2.models.surrogate import SQNSurrogate
from v2.models.ats import SQNConverted
from v2.models.stdp import SQNSTDP

def train_stdp_pretraining(model, dataset, device):
    """Unsupervised STDP Pre-training phase for the Backbone"""
    print("\n--- Starting Unsupervised STDP Pre-training ---")
    model.set_pretrain_mode(True)
    
    # Simple pass over all images
    for idx in range(len(dataset)):
        sample = dataset[idx]
        img = sample['image']
        
        # Format image for STDP
        img_transposed = np.transpose(img, (2, 0, 1))
        # Add batch dim
        img_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float().to(device) / 255.0
        
        # Forward pass triggers STDP weight updates internally
        model(img_tensor, None)
        
        if (idx + 1) % 10 == 0:
            print(f"Pre-training progress: {idx + 1}/{len(dataset)} images")
            
    model.set_pretrain_mode(False)
    print("--- STDP Pre-training Complete ---\n")

def run_rl_training(agent, dataset, epochs, epsilon_start=1.0, epsilon_min=0.1, decay_steps=10):
    """Standard DQN Training Loop"""
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_min) / decay_steps
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        epoch_loss = []
        epoch_reward = 0
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample['image']
            ground_truth = sample['box']
            
            history = [-1] * agent.history_size
            height, width, _ = image.shape
            current_mask = np.asarray([0, 0, width, height])
            
            step = 0
            done = False
            img_reward = 0
            
            while not done:
                current_mask, reward, done, history = agent.step(
                    image, history, current_mask, ground_truth, step, epsilon
                )
                
                loss = agent.train_step(batch_size=20)
                if loss > 0:
                    epoch_loss.append(loss)
                    
                img_reward += reward
                step += 1
                
            epoch_reward += img_reward
            
            if (idx + 1) % 10 == 0:
                print(f"Image {idx+1}: Reward = {img_reward}, Steps = {step}")
        
        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
        print(f"Epoch {epoch} Results: Avg Loss = {avg_loss:.4f}, Total Reward = {epoch_reward}, Epsilon = {epsilon:.2f}")
        
        if epsilon > epsilon_min:
            epsilon -= epsilon_decay

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Training (v2)")
    parser.add_argument('--method', type=str, choices=['surrogate', 'ats', 'stdp'], required=True, help="SNN method to use")
    parser.add_argument('--backbone', type=str, choices=['conv', 'vgg16'], default='conv', help="Feature extractor backbone")
    parser.add_argument('--target', type=str, default='mixing', help="Target class or 'mixing' for all")
    parser.add_argument('--num-samples', type=int, default=None, help="Number of samples to load from VOC")
    parser.add_argument('--simulate', type=int, default=10, help="Simulation timesteps for SNN")
    parser.add_argument('--epochs', type=int, default=10, help="Number of RL epochs")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    voc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'VOC2012')
    dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_samples)
    
    if len(dataset) == 0:
        print("No valid samples found. Exiting.")
        return

    # 2. Initialize Model
    if args.method == 'surrogate':
        model = SQNSurrogate(simulation_time=args.simulate, use_vgg16=(args.backbone == 'vgg16'))
    elif args.method == 'ats':
        model = SQNConverted(simulation_time=args.simulate, use_vgg16=(args.backbone == 'vgg16'))
    elif args.method == 'stdp':
        if args.backbone == 'vgg16':
            raise NotImplementedError("STDP method requires raw image input and cannot be used with a VGG16 backbone.")
        model = SQNSTDP()
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. Handle STDP Specifics
    if args.method == 'stdp':
        train_stdp_pretraining(model, dataset, device)
        # Re-initialize optimizer because STDP freezes some layers and we only want RL head to train
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 4. Initialize Agent
    agent = LocalizationAgent(model=model, optimizer=optimizer, device=device)
    
    # 5. Train RL
    print(f"Starting RL Loop using {args.method} mechanism...")
    run_rl_training(agent, dataset, epochs=args.epochs)
    
    # 6. ATS Conversion (if applicable)
    if args.method == 'ats':
        print("\n--- Converting ANN to SNN ---")
        model.convert_to_snn()

    # 7. Save Weights
    os.makedirs('weights', exist_ok=True)
    save_path = f"weights/{args.method}_{args.target}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
