import argparse
import torch
import os
import numpy as np
import csv

from data.voc_tfds import TFDSVOC2007TestDataset
from agents.localization_agent import LocalizationAgent
from models.surrogate import SQNSurrogate
from models.ats import SQNConverted

from helpers.tester import test_model
def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Testing (v2)")
    
    # Core Parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--method', type=str, choices=['surrogate', 'ats'], required=True, help="SNN method to evaluate: surrogate or ats")
    core_group.add_argument('--extractor', type=str, choices=['vgg16', 'resnet18', 'fusion', 'vit', 'efficientnet', 'mobilenet'], default='conv', help="Feature extractor backbone")
    core_group.add_argument('--target', type=str, default='mixing')
    core_group.add_argument('--num-samples', type=int, default=10, help="Test on 10 samples by default")
    core_group.add_argument('--voc-dir', type=str, default=None, help="Override default VOC2012 directory")
    
    # Agent Parameters
    agent_group = parser.add_argument_group('Agent Parameters')
    agent_group.add_argument('--replay', type=int, default=10, help="History size (history_size)")
    agent_group.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    
    # SNN Parameters
    snn_group = parser.add_argument_group('SNN Parameters')
    snn_group.add_argument('--simulate', type=int, default=10, help="Simulation timesteps for SNN")
    
    # System Parameters
    sys_group = parser.add_argument_group('System Parameters')
    sys_group.add_argument('--weights', type=str, default=None, help="Path to specific weights file")
    sys_group.add_argument('--logging-dir', type=str, default=None, help="Directory to save logs. If None, uses 'logs' folder.")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = TFDSVOC2007TestDataset(target_class=args.target, num_samples=args.num_samples)
    
    history_dim = 9 * args.replay
    if args.method == 'surrogate':
        model = SQNSurrogate(simulation_time=args.simulate, extractor_name=args.extractor, history_dim=history_dim)
    elif args.method == 'ats':
        model = SQNConverted(simulation_time=args.simulate, extractor_name=args.extractor, history_dim=history_dim)
        model.is_snn = True # Set to SNN mode for evaluation
        
    model = model.to(device)
    
    # Load weights
    weight_path = args.weights if args.weights else f"weights/{args.method}_{args.target}.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    elif args.weights:
        print(f"Error: Specified weights not found at {weight_path}")
        return
    else:
        print(f"Warning: Weights not found at {weight_path}. Evaluating with random weights.")
        
    # Agent wrapper (optimizer not needed for eval)
    agent = LocalizationAgent(model=model, device=device, history_size=args.replay, max_steps=args.max_steps)
    
    csv_file = f"test_{args.method}_{args.target}_{args.extractor}.csv"
    log_dir = args.logging_dir if args.logging_dir else "logs"
    test_model(agent, dataset, log_dir=log_dir, output_file=csv_file)

if __name__ == '__main__':
    main()
