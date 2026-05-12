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
# from models.stdp import SQNSTDP

from helpers.utils import get_optimizer, plot_training_results
from helpers.trainer import train_stdp_pretraining, run_rl_training

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Training (v2)")
    
    # Core Parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--method', type=str, choices=['surrogate', 'ats'], required=True, help="SNN method to use")
    core_group.add_argument('--extractor', type=str, choices=['vgg16', 'resnet18', 'fusion', 'vit', 'efficientnet', 'mobilenet'], default='conv', help="Feature extractor backbone")
    core_group.add_argument('--target', type=str, default='mixing', help="Target class or 'mixing' for all")
    core_group.add_argument('--num-samples', type=int, default=None, help="Number of samples to load from VOC")
    core_group.add_argument('--random', action='store_true', help="Random sample from dataset")
    core_group.add_argument('--voc-dir', type=str, default=None, help="Override default VOC2012 directory")
    
    # RL/Agent Parameters
    rl_group = parser.add_argument_group('RL/Agent Parameters')
    rl_group.add_argument('--algo', type=str, choices=['dqn', 'double', 'dueling'], default='dqn', help="RL algorithm to use")
    rl_group.add_argument('--gamma', type=float, default=0.99, help="Discount factor for future rewards")
    rl_group.add_argument('--epochs', type=int, default=10, help="Number of RL epochs")
    rl_group.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    rl_group.add_argument('--alpha', type=float, default=0.1, help="Mask transformation rate")
    rl_group.add_argument('--nu', type=float, default=3.0, help="Trigger reward weight")
    rl_group.add_argument('--threshold', type=float, default=0.5, help="IoU threshold for trigger reward")
    rl_group.add_argument('--replay', type=int, default=10, help="History size (history_size)")
    rl_group.add_argument('--target-update', type=int, default=1, help="Epochs between target network updates")
    rl_group.add_argument('--loss-fn', type=str, choices=['mse', 'huber', 'smooth_l1'], default='huber', help="Loss function for RL")
    rl_group.add_argument('--use-cache', action='store_true', default=True, help="Use feature caching during training")
    
    # SNN Parameters
    snn_group = parser.add_argument_group('SNN Parameters')
    snn_group.add_argument('--simulate', type=int, default=10, help="Simulation timesteps for SNN")
    # snn_group.add_argument('--stdp-epochs', type=int, default=3, help="Number of STDP pretraining epochs")
    
    # Optimizer/Training Parameters
    train_group = parser.add_argument_group('Training/Optimizer Parameters')
    train_group.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'rmsprop', 'sgd', 'radam'], default='adam', help="Optimizer to use")
    train_group.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    train_group.add_argument('--weight-decay', type=float, default=0.0, help="Weight decay for optimizer")
    train_group.add_argument('--clip-grad', type=float, default=1.0, help="Gradient clipping norm")
    train_group.add_argument('--batch-size', type=int, default=20, help="Batch size for training")
    train_group.add_argument('--early-stop', type=int, default=0, help="Early stopping if no improvement for N epochs")
    train_group.add_argument('--validation', type=str, choices=['none', 'loss', 'iou'], default='none', help="Validation metric to use for saving best model")
    train_group.add_argument('--val-ratio', type=float, default=0.2, help="Ratio of validation samples when validation is enabled")
    
    # Logging and Saving
    log_group = parser.add_argument_group('Logging and Saving')
    log_group.add_argument('--logging-dir', type=str, default=None, help="Directory to save logs. If None, uses 'logs' folder.")
    log_group.add_argument('--save', type=str, choices = ["best", "last", "epoch", "none"], default="last", help="Save model mode")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    voc_dir = args.voc_dir if args.voc_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    
    val_dataset = None
    if args.validation != 'none':
        train_samples = int(args.num_samples * (1 - args.val_ratio)) if args.num_samples else None
        val_samples = int(args.num_samples * args.val_ratio) if args.num_samples else None
        
        dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=train_samples, split='train', use_random=args.random)
        val_dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=val_samples, split='val', use_random=args.random)
        print(f"Validation enabled. Train samples: {len(dataset)}, Val samples: {len(val_dataset)}")
    else:
        dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_samples, split='train', use_random=args.random)
    
    if len(dataset) == 0:
        print("No valid samples found. Exiting.")
        return

    # 2. Initialize Model
    history_dim = 9 * args.replay
    is_dueling = (args.algo == 'dueling')
    
    if args.method == 'surrogate':
        model = SQNSurrogate(simulation_time=args.simulate, extractor_name=args.extractor, history_dim=history_dim, dueling=is_dueling)
    elif args.method == 'ats':
        model = SQNConverted(simulation_time=args.simulate, extractor_name=args.extractor, history_dim=history_dim, dueling=is_dueling)
    # elif args.method == 'stdp':
    #     if args.extractor == 'vgg16':
    #         raise NotImplementedError("STDP method requires raw image input and cannot be used with a VGG16 backbone.")
    #     model = SQNSTDP(history_dim=history_dim, dueling=is_dueling)
        
    model = model.to(device)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    
    # # 3. Handle STDP Specifics
    # if args.method == 'stdp':
    #     train_stdp_pretraining(model, dataset, device, stdp_epochs=args.stdp_epochs)
    #     # Re-initialize optimizer because STDP freezes some layers and we only want RL head to train
    #     optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # 4. Initialize Engine
    from backbone.engine import DQNEngine, DoubleDQNEngine
    if args.algo == 'double':
        engine = DoubleDQNEngine(model, gamma=args.gamma, use_target_net=True)
    elif args.algo == 'dueling':
        engine = DQNEngine(model, gamma=args.gamma, use_target_net=True)
    else:
        # standard dqn
        engine = DQNEngine(model, gamma=args.gamma, use_target_net=True)

    # 5. Initialize Agent
    agent = LocalizationAgent(
        model=model,
        engine=engine,
        optimizer=optimizer, 
        device=device, 
        gamma=args.gamma,
        nu=args.nu,
        threshold=args.threshold,
        clip_grad=args.clip_grad,
        loss_fn=args.loss_fn,
        max_steps=args.max_steps,
        alpha=args.alpha,
        history_size=args.replay,
        use_cache=args.use_cache
    )
    
    # 6. Train RL
    print(f"Starting RL Loop using {args.method} mechanism with {args.algo.upper()}...")
    save_path = f"weights/{args.method}_{args.target}.pth"
    losses, epsilons = run_rl_training(
        agent, dataset, epochs=args.epochs, 
        early_stop_patience=args.early_stop,
        save_mode=args.save,
        save_path=save_path,
        batch_size=args.batch_size,
        target_update=args.target_update,
        val_dataset=val_dataset,
        validation_mode=args.validation
    )
    
    log_dir = args.logging_dir if args.logging_dir else "logs"
    plot_training_results(losses, epsilons, args.method, args.target, log_dir=log_dir)
    
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
