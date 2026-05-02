import argparse
import torch
import os
import numpy as np
import csv

from data.voc import VOCDataset
from agents.localization_agent import LocalizationAgent
from models.surrogate import SQNSurrogate
from models.ats import SQNConverted
from models.stdp import SQNSTDP

from helpers.tester import test_model
def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Testing (v2)")
    
    # Core Parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--method', type=str, choices=['surrogate', 'ats', 'stdp'], required=True)
    core_group.add_argument('--backbone', type=str, choices=['conv', 'vgg16', 'resnet18'], default='conv')
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
    sys_group.add_argument('--logging', action='store_true', help="Log metrics to CSV")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    voc_dir = args.voc_dir if args.voc_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_samples, split="val")
    
    history_dim = 9 * args.replay
    if args.method == 'surrogate':
        model = SQNSurrogate(simulation_time=args.simulate, backbone_name=args.backbone, history_dim=history_dim)
    elif args.method == 'ats':
        model = SQNConverted(simulation_time=args.simulate, backbone_name=args.backbone, history_dim=history_dim)
        model.is_snn = True # Set to SNN mode for evaluation
    elif args.method == 'stdp':
        if args.backbone == 'vgg16':
            raise ValueError("STDP method requires raw image input and cannot be used with a VGG16 backbone.")
        model = SQNSTDP(history_dim=history_dim)
        model.set_pretrain_mode(False) # Ensure RL head is active
        
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
    
    csv_file = f"test_{args.method}_{args.target}_{args.backbone}.csv"
    test_model(agent, dataset, logging=args.logging, output_file=csv_file)

if __name__ == '__main__':
    main()
