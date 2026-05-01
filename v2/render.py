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


def render_predictions(agent, samples, save_dir=None):
    print(f"\n--- Rendering Visualizations for {len(samples)} samples ---")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Rendered images will be saved to {save_dir}")

    for idx, sample in enumerate(samples):
        image = sample['image']
        ground_truth = sample.get('box')
        
        history = [-1] * agent.history_size
        height, width, _ = image.shape
        current_mask = np.asarray([0, 0, width, height])
        
        step = 0
        done = False
        masks = []
        
        # Simulation Loop (greedy policy)
        while not done and step < agent.max_steps:
            img_tensor, hist_tensor = agent.feature_extract(image, history, width, height, current_mask)
            
            agent.model.eval()
            with torch.no_grad():
                q_values = agent.model(img_tensor.to(agent.device), hist_tensor.to(agent.device))
                action = torch.argmax(q_values).item()
                
            history = history[1:] + [action]
            
            if action == 8:
                done = True
                new_mask = current_mask
            else:
                new_mask = agent.compute_mask(action, current_mask)
                
            masks.append(new_mask)
            current_mask = new_mask
            step += 1
            
        final_mask = current_mask
        
        # Visualization
        vis_img = image.copy()
        
        # Draw ground truth (Green) - only if available
        if ground_truth is not None and np.any(ground_truth > 0):
            cv2.rectangle(vis_img, (int(ground_truth[0]), int(ground_truth[1])), 
                          (int(ground_truth[2]), int(ground_truth[3])), (0, 255, 0), 2)
        
        # Draw intermediate predictions (Blue, thin)
        for m in masks[:-1]:
             cv2.rectangle(vis_img, (int(m[0]), int(m[1])), 
                      (int(m[2]), int(m[3])), (0, 0, 255), 1)

        # Draw final prediction (Red)
        cv2.rectangle(vis_img, (int(final_mask[0]), int(final_mask[1])), 
                      (int(final_mask[2]), int(final_mask[3])), (255, 0, 0), 2)
                      
        if save_dir:
            save_path = os.path.join(save_dir, f"sample_{idx+1}.png")
            # Save as BGR for cv2
            cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"Saved visualization to {save_path}")

        filename = sample.get('filename', f"sample_{idx+1}")
        print(f"Sample {idx+1} ({filename}): Displaying result...")
        plt.figure(figsize=(8, 6))
        plt.imshow(vis_img)
        plt.title(f"Sample {idx+1} - Prediction")
        plt.axis('off')
        plt.show()
    print("--- Visualization Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Visualization (v2)")
    parser.add_argument('--method', type=str, choices=['surrogate', 'ats', 'stdp'], required=True)
    parser.add_argument('--backbone', type=str, choices=['conv', 'vgg16', 'resnet18'], default='conv')
    parser.add_argument('--target', type=str, default='mixing')
    parser.add_argument('--image-path', type=str, default=None, help="Path to specific image file")
    parser.add_argument('--num-images', type=int, default=5, help="Number of images if no path provided")
    parser.add_argument('--simulate', type=int, default=10)
    parser.add_argument('--replay', type=int, default=10, help="History size (history_size)")
    parser.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    parser.add_argument('--weights', type=str, default=None, help="Path to specific weights file")
    parser.add_argument('--voc-dir', type=str, default=None, help="Override default VOC2012 directory")
    parser.add_argument('--save', action='store_true', help="Save rendered images to disk")
    parser.add_argument('--save-dir', type=str, default=None, help="Directory to save rendered images")
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
