import argparse
import torch
import os
import sys

# Ensure imports work by adding the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.data.voc import VOCDataset
from baseline.utils.agent import Agent
from v2.helpers.trainer import run_rl_training

def main():
    parser = argparse.ArgumentParser(description="Baseline Agent Training with v2 Interface")
    
    # Core Parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--target', type=str, default='mixing', help="Target class or 'mixing' for all")
    core_group.add_argument('--extractor', type=str, choices=['vgg16', 'resnet18', 'vit', 'efficientnet', 'mobilenet'], default='vgg16', help="Feature extractor backbone")
    core_group.add_argument('--num-samples', type=int, default=None, help="Number of samples to load from VOC")
    core_group.add_argument('--random', action='store_true', help="Random sample from dataset")
    core_group.add_argument('--voc-dir', type=str, default=None, help="Override default VOC2012 directory")
    
    # RL/Agent Parameters
    rl_group = parser.add_argument_group('RL/Agent Parameters')
    rl_group.add_argument('--epochs', type=int, default=10, help="Number of RL epochs")
    rl_group.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    rl_group.add_argument('--alpha', type=float, default=0.1, help="Mask transformation rate")
    rl_group.add_argument('--nu', type=float, default=3.0, help="Trigger reward weight")
    rl_group.add_argument('--threshold', type=float, default=0.5, help="IoU threshold for trigger reward")
    rl_group.add_argument('--target-update', type=int, default=1, help="Epochs between target network updates")
    rl_group.add_argument('--use-cache', action='store_true', default=True, help="Use feature caching during training")
    
    # Optimizer/Training Parameters
    train_group = parser.add_argument_group('Training/Optimizer Parameters')
    train_group.add_argument('--batch-size', type=int, default=20, help="Batch size for training")
    train_group.add_argument('--early-stop', type=int, default=0, help="Early stopping if no improvement for N epochs")
    train_group.add_argument('--validation', type=str, choices=['none', 'loss', 'iou'], default='none', help="Validation metric to use for saving best model")
    train_group.add_argument('--val-ratio', type=float, default=0.2, help="Ratio of validation samples when validation is enabled")
    
    # Logging and Saving
    log_group = parser.add_argument_group('Logging and Saving')
    log_group.add_argument('--save', type=str, choices=["best", "last", "epoch", "none"], default="last", help="Save model mode")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    voc_dir = args.voc_dir if args.voc_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    
    val_dataset = None
    if args.validation != 'none':
        train_samples = int(args.num_samples * (1 - args.val_ratio)) if args.num_samples else None
        val_samples = int(args.num_samples * args.val_ratio) if args.num_samples else None
        
        dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=train_samples, split='train', use_random=args.random)
        val_dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=val_samples, split='val', use_random=args.random)
    else:
        dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_samples, split='train', use_random=args.random)
        
    if len(dataset) == 0:
        print("No valid samples found. Exiting.")
        return
        
    agent = Agent(
        classe=args.target,
        alpha=args.alpha,
        nu=args.nu,
        threshold=args.threshold,
        max_steps=args.max_steps,
        device=device,
        extractor_name=args.extractor,
        use_cache=args.use_cache
    )
    
    os.makedirs('baseline/weights', exist_ok=True)
    save_path = f"baseline/weights/baseline_{args.target}.pth"
    print(f"Starting Baseline RL Loop for target {args.target}...")
    
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
    
    if args.save == "last":
        torch.save(agent.model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")
    elif args.save == "best":
        print(f"Best model was saved to {save_path}")
    elif args.save == "epoch":
        print(f"Epoch models were saved in weights directory.")
    else:
        print("Model saving skipped (none).")

if __name__ == '__main__':
    main()
