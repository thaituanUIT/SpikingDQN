import argparse
import torch
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data.voc import VOCDataset
from agents.localization_agent import LocalizationAgent
from models.spikingjelly_model import SQNJelly
from models.stdp_jelly_model import SQNSTDPJelly

def get_optimizer(model, opt_name, lr):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if opt_name == 'adam':
        return optim.Adam(parameters, lr=lr)
    elif opt_name == 'adamw':
        return optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, alpha=0.99, eps=1e-8)
    elif opt_name == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        return optim.Adam(parameters, lr=lr)

def train_stdp_pretraining(model, dataset, device):
    """Unsupervised STDP Pre-training phase for the Backbone"""
    print("\n--- Starting Unsupervised STDP Pre-training ---")
    model.set_pretrain_mode(True)
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        img = sample['image']
        img_transposed = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float().to(device) / 255.0
        
        # Forward pass triggers STDP weight updates internally in SQNSTDPJelly
        model(img_tensor, torch.zeros(1, model.history_dim, device=device))
        
        if (idx + 1) % 10 == 0:
            print(f"Pre-training progress: {idx + 1}/{len(dataset)} images")
            
    model.set_pretrain_mode(False)
    print("--- STDP Pre-training Complete ---\n")

def run_rl_training(agent, dataset, epochs, epsilon_start=1.0, epsilon_min=0.1, decay_steps=10):
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_min) / decay_steps
    
    history_loss = []
    history_epsilon = []
    
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
        
        history_loss.append(avg_loss)
        history_epsilon.append(epsilon)
        
        if epsilon > epsilon_min:
            epsilon -= epsilon_decay

    return history_loss, history_epsilon

def plot_training_results(losses, epsilons, method, target):
    os.makedirs('v3/logs', exist_ok=True)
    df = pd.DataFrame({'epoch': range(1, len(losses) + 1), 'loss': losses, 'epsilon': epsilons})
    csv_path = f"v3/logs/{method}_{target}_training_log.csv"
    df.to_csv(csv_path, index=False)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, len(losses) + 1), losses, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(range(1, len(epsilons) + 1), epsilons, color=color, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Training Metrics (v3 {method} - {target})")
    fig.tight_layout()
    plot_path = f"v3/logs/{method}_{target}_metrics.png"
    plt.savefig(plot_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Training (v3 - SpikingJelly)")
    parser.add_argument('--method', type=str, default='jelly', choices=['jelly', 'stdp_jelly'], help="SNN method to use")
    parser.add_argument('--backbone', type=str, choices=['conv', 'vgg16', 'resnet18'], default='conv', help="Feature extractor backbone")
    parser.add_argument('--target', type=str, default='mixing', help="Target class or 'mixing' for all")
    parser.add_argument('--num-samples', type=int, default=None, help="Number of samples to load from VOC")
    parser.add_argument('--simulate', type=int, default=10, help="Simulation timesteps for SNN")
    parser.add_argument('--epochs', type=int, default=10, help="Number of RL epochs")
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'rmsprop', 'sgd'], default='adam', help="Optimizer to use")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--clip-grad', type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument('--logging', action='store_true', help="Enable logging")
    parser.add_argument('--voc-dir', type=str, default=None, help="Override default VOC2012 directory")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    voc_dir = args.voc_dir if args.voc_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VOC2012')
    dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_samples)
    
    if len(dataset) == 0:
        print("No valid samples found. Exiting.")
        return

    # Initialize Model
    if args.method == 'jelly':
        model = SQNJelly(simulation_time=args.simulate, backbone_name=args.backbone)
    elif args.method == 'stdp_jelly':
        if args.backbone != 'conv':
            raise ValueError("STDP Jelly method currently only supports the 'conv' backbone.")
        model = SQNSTDPJelly(simulation_time=args.simulate)
        
    model = model.to(device)
    
    # Handle STDP Pre-training
    if args.method == 'stdp_jelly' and args.epochs > 0:
        train_stdp_pretraining(model, dataset, device)

    optimizer = get_optimizer(model, args.optimizer, args.lr)
    
    # Initialize Agent
    agent = LocalizationAgent(model=model, optimizer=optimizer, device=device, clip_grad=args.clip_grad)
    
    # Train RL
    print(f"Starting RL Loop using SpikingJelly ({args.method})...")
    losses, epsilons = run_rl_training(agent, dataset, epochs=args.epochs)
    
    if args.logging:
        plot_training_results(losses, epsilons, args.method, args.target)
    
    # Save Weights
    os.makedirs('v3/weights', exist_ok=True)
    save_path = f"v3/weights/{args.method}_{args.target}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
