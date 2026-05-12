import argparse
import torch
import os
import sys

# Ensure imports work by adding the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.data.voc_tfds import TFDSVOC2007TestDataset
from baseline.utils.agent import Agent
from v2.helpers.tester import test_model

def main():
    parser = argparse.ArgumentParser(description="Baseline Agent Testing with v2 Interface")
    
    # Core Parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--target', type=str, default='mixing', help="Target class or 'mixing' for all")
    core_group.add_argument('--extractor', type=str, choices=['vgg16', 'resnet18', 'vit', 'efficientnet', 'mobilenet'], default='vgg16', help="Feature extractor backbone")
    core_group.add_argument('--num-samples', type=int, default=10, help="Test on 10 samples by default")
    
    # Agent Parameters
    agent_group = parser.add_argument_group('Agent Parameters')
    agent_group.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    agent_group.add_argument('--alpha', type=float, default=0.1, help="Mask transformation rate")
    agent_group.add_argument('--nu', type=float, default=3.0, help="Trigger reward weight")
    agent_group.add_argument('--threshold', type=float, default=0.5, help="IoU threshold for trigger reward")
    
    # System Parameters
    sys_group = parser.add_argument_group('System Parameters')
    sys_group.add_argument('--weights', type=str, default=None, help="Path to specific weights file")
    sys_group.add_argument('--logging', action='store_true', help="Log metrics to CSV")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = TFDSVOC2007TestDataset(target_class=args.target, num_samples=args.num_samples)
    
    agent = Agent(
        classe=args.target,
        alpha=args.alpha,
        nu=args.nu,
        threshold=args.threshold,
        max_steps=args.max_steps,
        device=device,
        extractor_name=args.extractor
    )
    
    # Load weights
    weight_path = args.weights if args.weights else f"baseline/weights/baseline_{args.target}.pth"
    if os.path.exists(weight_path):
        agent.model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    elif args.weights:
        print(f"Error: Specified weights not found at {weight_path}")
        return
    else:
        print(f"Warning: Weights not found at {weight_path}. Evaluating with random weights.")
        
    csv_file = f"test_baseline_{args.target}.csv"
    test_model(agent, dataset, logging=args.logging, output_file=csv_file)

if __name__ == '__main__':
    main()
