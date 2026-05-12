import argparse
import torch
import os
import sys
import cv2
import numpy as np

# Ensure imports work by adding the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.data.voc import VOCDataset
from baseline.utils.agent import Agent
from v2.helpers.renderer import render_predictions

def main():
    parser = argparse.ArgumentParser(description="Baseline Agent Rendering with v2 Interface")
    
    # Core Parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--target', type=str, default='mixing', help="Target class or 'mixing' for all")
    core_group.add_argument('--image-path', type=str, default=None, help="Path to specific image file")
    core_group.add_argument('--num-images', type=int, default=5, help="Number of images if no path provided")
    core_group.add_argument('--extractor', type=str, choices=['vgg16', 'resnet18', 'vit', 'efficientnet', 'mobilenet'], default='vgg16', help="Feature extractor backbone")
    core_group.add_argument('--voc-dir', type=str, default=None, help="Override default VOC2012 directory")
    
    # Agent Parameters
    agent_group = parser.add_argument_group('Agent Parameters')
    agent_group.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    agent_group.add_argument('--alpha', type=float, default=0.1, help="Mask transformation rate")
    agent_group.add_argument('--nu', type=float, default=3.0, help="Trigger reward weight")
    agent_group.add_argument('--threshold', type=float, default=0.5, help="IoU threshold for trigger reward")
    
    # System Parameters
    sys_group = parser.add_argument_group('System Parameters')
    sys_group.add_argument('--weights', type=str, default=None, help="Path to specific weights file")
    sys_group.add_argument('--save', action='store_true', help="Save rendered images to disk")
    sys_group.add_argument('--save-dir', type=str, default=None, help="Directory to save rendered images")
    
    args = parser.parse_args()
    
    if args.save and not args.save_dir:
        args.save_dir = f"baseline/renders/baseline_{args.target}"
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    else:
        status = "Error" if args.weights else "Warning"
        print(f"{status}: Weights not found at {weight_path}. Cannot render without trained weights.")
        return
        
    render_predictions(agent, samples, save_dir=args.save_dir)

if __name__ == '__main__':
    main()
