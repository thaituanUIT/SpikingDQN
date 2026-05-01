import argparse
import torch
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data.voc import VOCDataset
from agents.localization_agent import LocalizationAgent
from models.surrogate import SQNSurrogate
from models.ats import SQNConverted
from models.stdp import SQNSTDP

def get_optimizer(model, opt_name, lr, weight_decay=0.0):
    """Factory function to create the requested optimizer"""
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    if opt_name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay if weight_decay > 0 else 0.01)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, alpha=0.99, eps=1e-8, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == 'radam':
        return optim.RAdam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

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

def run_rl_training(agent, dataset, epochs, epsilon_start=1.0, epsilon_min=0.1, decay_steps=10, early_stop_patience=0, save_mode="none", save_path="weights/best_model.pth", batch_size=20):
    """Standard DQN Training Loop"""
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_min) / decay_steps
    
    # Track logs
    history_loss = []
    history_epsilon = []
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Ensure weights directory exists if saving
    if save_mode != "none":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
                
                loss = agent.train_step(batch_size=batch_size)
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
        
        # Save model based on mode
        if save_mode == "best" and avg_loss < best_loss:
            torch.save(agent.model.state_dict(), save_path)
            print(f"New best model saved with Loss: {avg_loss:.4f}")
        elif save_mode == "epoch":
            epoch_save_path = save_path.replace(".pth", f"_epoch_{epoch}.pth")
            torch.save(agent.model.state_dict(), epoch_save_path)
            print(f"Model saved for epoch {epoch} to {epoch_save_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epsilon > epsilon_min:
            epsilon -= epsilon_decay
            
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. No improvement in Avg Loss for {early_stop_patience} epochs.")
            break

    return history_loss, history_epsilon

def plot_training_results(losses, epsilons, method, target):
    """Save training metrics to CSV and plot them"""
    os.makedirs('logs', exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame({
        'epoch': range(1, len(losses) + 1),
        'loss': losses,
        'epsilon': epsilons
    })
    csv_path = f"logs/{method}_{target}_training_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training logs saved to {csv_path}")
    
    # Plotting
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

    plt.title(f"Training Metrics ({method} - {target})")
    fig.tight_layout()
    
    plot_path = f"logs/{method}_{target}_metrics.png"
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Training (v2)")
    parser.add_argument('--method', type=str, choices=['surrogate', 'ats', 'stdp'], required=True, help="SNN method to use")
    parser.add_argument('--backbone', type=str, choices=['conv', 'vgg16', 'resnet18'], default='conv', help="Feature extractor backbone")
    parser.add_argument('--target', type=str, default='mixing', help="Target class or 'mixing' for all")
    parser.add_argument('--num-samples', type=int, default=None, help="Number of samples to load from VOC")
    parser.add_argument('--simulate', type=int, default=10, help="Simulation timesteps for SNN")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument('--epochs', type=int, default=10, help="Number of RL epochs")
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'rmsprop', 'sgd', 'radam'], default='adam', help="Optimizer to use")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument('--clip-grad', type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument('--early-stop', type=int, default=0, help="Early stopping if no improvement for N epochs")
    parser.add_argument('--logging', action='store_true', help="Enable logging")
    parser.add_argument('--save', type=str, choices = ["best", "last", "epoch", "none"], default="last", help="Save model mode")
    parser.add_argument('--loss-fn', type=str, choices=['mse', 'huber', 'smooth_l1'], default='huber', help="Loss function for RL")
    parser.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    parser.add_argument('--alpha', type=float, default=0.1, help="Mask transformation rate")
    parser.add_argument('--nu', type=float, default=3.0, help="Trigger reward weight")
    parser.add_argument('--threshold', type=float, default=0.5, help="IoU threshold for trigger reward")
    parser.add_argument('--batch-size', type=int, default=20, help="Batch size for training")
    parser.add_argument('--replay', type=int, default=10, help="History size (history_size)")
    parser.add_argument('--voc-dir', type=str, default=None, help="Override default VOC2012 directory")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    voc_dir = args.voc_dir if args.voc_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_samples)
    
    if len(dataset) == 0:
        print("No valid samples found. Exiting.")
        return

    # 2. Initialize Model
    history_dim = 9 * args.replay
    if args.method == 'surrogate':
        model = SQNSurrogate(simulation_time=args.simulate, backbone_name=args.backbone, history_dim=history_dim)
    elif args.method == 'ats':
        model = SQNConverted(simulation_time=args.simulate, backbone_name=args.backbone, history_dim=history_dim)
    elif args.method == 'stdp':
        if args.backbone == 'vgg16':
            raise NotImplementedError("STDP method requires raw image input and cannot be used with a VGG16 backbone.")
        model = SQNSTDP(history_dim=history_dim)
        
    model = model.to(device)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    
    # 3. Handle STDP Specifics
    if args.method == 'stdp':
        train_stdp_pretraining(model, dataset, device)
        # Re-initialize optimizer because STDP freezes some layers and we only want RL head to train
        optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # 4. Initialize Agent
    agent = LocalizationAgent(
        model=model, 
        optimizer=optimizer, 
        device=device, 
        gamma=args.gamma,
        nu=args.nu,
        threshold=args.threshold,
        clip_grad=args.clip_grad,
        loss_fn=args.loss_fn,
        max_steps=args.max_steps,
        alpha=args.alpha,
        history_size=args.replay
    )
    
    # 5. Train RL
    print(f"Starting RL Loop using {args.method} mechanism...")
    save_path = f"weights/{args.method}_{args.target}.pth"
    losses, epsilons = run_rl_training(
        agent, dataset, epochs=args.epochs, 
        early_stop_patience=args.early_stop,
        save_mode=args.save,
        save_path=save_path,
        batch_size=args.batch_size
    )
    
    if args.logging:
        plot_training_results(losses, epsilons, args.method, args.target)
    
    # 6. ATS Conversion (if applicable)
    if args.method == 'ats':
        print("\n--- Converting ANN to SNN ---")
        model.convert_to_snn()

    # 7. Save Weights
    if args.save == "last":
        os.makedirs('weights', exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")
    elif args.save == "best":
        print(f"Best model was saved to {save_path}")
    elif args.save == "epoch":
        print(f"Epoch models were saved in weights directory.")
    else:
        print("Model saving skipped (none).")

if __name__ == '__main__':
    main()
