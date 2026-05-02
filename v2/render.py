import argparse
import torch
import os
import numpy as np
import cv2

from data.voc import VOCDataset
from agents.localization_agent import LocalizationAgent
from models.surrogate import SQNSurrogate
from models.ats import SQNConverted
from models.stdp import SQNSTDP

import matplotlib.pyplot as plt


from helpers.renderer import render_predictions

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Visualization (v2)")
    
    # Core Parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--method', type=str, choices=['surrogate', 'ats', 'stdp'], required=True)
    core_group.add_argument('--backbone', type=str, choices=['conv', 'vgg16', 'resnet18'], default='conv')
    core_group.add_argument('--target', type=str, default='mixing')
    core_group.add_argument('--image-path', type=str, default=None, help="Path to specific image file")
    core_group.add_argument('--num-images', type=int, default=5, help="Number of images if no path provided")
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
    sys_group.add_argument('--save', action='store_true', help="Save rendered images to disk")
    sys_group.add_argument('--save-dir', type=str, default=None, help="Directory to save rendered images")
    args = parser.parse_args()

    if args.save and not args.save_dir:
        args.save_dir = f"renders/{args.method}_{args.target}"

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image path {args.image_path} not found.")
            return
        img_bgr = cv2.imread(args.image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        samples = [{
            'image': img_rgb,
            'box': None,
            'filename': os.path.basename(args.image_path)
        }]
    else:
        voc_dir = args.voc_dir if args.voc_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
        dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_images)
        samples = [dataset[i] for i in range(len(dataset))]
    
    history_dim = 9 * args.replay
    if args.method == 'surrogate':
        model = SQNSurrogate(simulation_time=args.simulate, backbone_name=args.backbone, history_dim=history_dim)
    elif args.method == 'ats':
        model = SQNConverted(simulation_time=args.simulate, backbone_name=args.backbone, history_dim=history_dim)
        model.is_snn = True
    elif args.method == 'stdp':
        if args.backbone == 'vgg16':
            raise ValueError("STDP method requires raw image input and cannot be used with a VGG16 backbone.")
        model = SQNSTDP(history_dim=history_dim)
        model.set_pretrain_mode(False)
        
    model = model.to(device)
    
    weight_path = args.weights if args.weights else f"weights/{args.method}_{args.target}.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        status = "Error" if args.weights else "Warning"
        print(f"{status}: Weights not found at {weight_path}. Cannot render without trained weights.")
        return
        
    agent = LocalizationAgent(model=model, device=device, history_size=args.replay, max_steps=args.max_steps)
    render_predictions(agent, samples, save_dir=args.save_dir)



if __name__ == '__main__':
    main()
