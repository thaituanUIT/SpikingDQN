import argparse
import torch
import os
import numpy as np
import csv

from data.voc import VOCDataset
from agents.localization_agent import LocalizationAgent
from models.spikingjelly_model import SQNJelly
from models.stdp_jelly_model import SQNSTDPJelly

def test_model(agent, dataset, logging=False, output_file='test_results.csv'):
    print(f"\n--- Starting Evaluation on {len(dataset)} samples ---")
    
    total_iou = []
    total_steps = []
    log_data = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample['image']
        ground_truth = sample['box']
        
        history = [-1] * agent.history_size
        height, width, _ = image.shape
        current_mask = np.asarray([0, 0, width, height])
        
        step = 0
        done = False
        
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
                
            current_mask = new_mask
            step += 1
            
        final_mask = current_mask
        iou = agent.compute_iou(final_mask, ground_truth)
        total_iou.append(iou)
        total_steps.append(step)
        
        log_data.append({
            'Image_ID': idx+1,
            'Ground_Truth': tuple(ground_truth),
            'Prediction': tuple(final_mask),
            'Steps': step,
            'IoU': iou
        })
        
        print(f"Sample {idx+1}: IoU = {iou:.4f}, Steps taken = {step}")
        
    avg_iou = np.mean(total_iou) if total_iou else 0
    avg_steps = np.mean(total_steps) if total_steps else 0
    
    acc_03 = sum(1 for iou in total_iou if iou >= 0.3) / len(total_iou) if total_iou else 0
    acc_05 = sum(1 for iou in total_iou if iou >= 0.5) / len(total_iou) if total_iou else 0
    
    print(f"\n--- Evaluation Metrics ---")
    print(f"Average Final IoU: {avg_iou:.4f}")
    print(f"Average Steps Taken: {avg_steps:.2f}")
    print(f"Localization Accuracy (IoU >= 0.3): {acc_03*100:.2f}%")
    print(f"Localization Accuracy (IoU >= 0.5): {acc_05*100:.2f}%")
    
    if logging:
        os.makedirs('v3/logs', exist_ok=True)
        csv_path = os.path.join('v3/logs', output_file)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Image_ID', 'Ground_Truth', 'Prediction', 'Steps', 'IoU'])
            writer.writeheader()
            writer.writerows(log_data)
        print(f"-> Detailed metrics logged to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Testing (v3 - SpikingJelly)")
    parser.add_argument('--method', type=str, default='jelly', choices=['jelly', 'stdp_jelly'])
    parser.add_argument('--backbone', type=str, choices=['conv', 'vgg16', 'resnet18', 'fusion'], default='conv')
    parser.add_argument('--target', type=str, default='mixing')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--simulate', type=int, default=10)
    parser.add_argument('--replay', type=int, default=10, help="History size")
    parser.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    parser.add_argument('--logging', action='store_true', help="Log metrics to CSV")
    parser.add_argument('--voc-dir', type=str, default=None, help="Override default VOC2012 directory")
    parser.add_argument('--weights', type=str, default=None, help="Path to specific weights file")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    voc_dir = args.voc_dir if args.voc_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_samples, split="val")
    
    history_dim = 9 * args.replay
    if args.method == 'jelly':
        model = SQNJelly(simulation_time=args.simulate, backbone_name=args.backbone, history_dim=history_dim)
    elif args.method == 'stdp_jelly':
        model = SQNSTDPJelly(simulation_time=args.simulate, history_dim=history_dim)
    model = model.to(device)
    
    weight_path = args.weights if args.weights else f"v3/weights/{args.method}_{args.target}.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    elif args.weights:
        print(f"Error: Specified weights not found at {weight_path}")
        return
    else:
        print(f"Warning: Weights not found at {weight_path}. Evaluating with random weights.")
        
    agent = LocalizationAgent(model=model, device=device, history_size=args.replay, max_steps=args.max_steps)
    
    csv_file = f"test_{args.method}_{args.target}_{args.backbone}.csv"
    test_model(agent, dataset, logging=args.logging, output_file=csv_file)

if __name__ == '__main__':
    main()
