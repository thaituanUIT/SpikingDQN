import argparse
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

from v2.data.voc import VOCDataset
from v2.agents.localization_agent import LocalizationAgent
from v2.models.surrogate import SQNSurrogate
from v2.models.ats import SQNConverted
from v2.models.stdp import SQNSTDP

def test_model(agent, dataset, render=True, logging=False, output_file='test_results.csv'):
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
        masks = []
        
        # Test Loop (greedy policy)
        while not done and step < agent.max_steps:
            # feature extraction
            img_tensor, hist_tensor = agent.feature_extract(image, history, width, height, current_mask)
            
            # Predict action (epsilon = 0.0 for greedy)
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
        
        if render and idx < 5: # Render first 5
            vis_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV uses BGR for lines
            
            # Draw ground truth (Green)
            cv2.rectangle(vis_img, (int(ground_truth[0]), int(ground_truth[1])), 
                          (int(ground_truth[2]), int(ground_truth[3])), (0, 255, 0), 2)
            
            # Draw final prediction (Red)
            cv2.rectangle(vis_img, (int(final_mask[0]), int(final_mask[1])), 
                          (int(final_mask[2]), int(final_mask[3])), (0, 0, 255), 2)
            
            # Draw intermediate predictions (Blue, thin)
            for m in masks[:-1]:
                 cv2.rectangle(vis_img, (int(m[0]), int(m[1])), 
                          (int(m[2]), int(m[3])), (255, 0, 0), 1)
                          
            cv2.imshow(f"Result {idx}", vis_img)
            cv2.waitKey(0)
            
    if render:
        cv2.destroyAllWindows()
        
    avg_iou = np.mean(total_iou) if total_iou else 0
    avg_steps = np.mean(total_steps) if total_steps else 0
    
    acc_03 = sum(1 for iou in total_iou if iou >= 0.3) / len(total_iou) if total_iou else 0
    acc_05 = sum(1 for iou in total_iou if iou >= 0.5) / len(total_iou) if total_iou else 0
    acc_07 = sum(1 for iou in total_iou if iou >= 0.7) / len(total_iou) if total_iou else 0
    
    print(f"\n--- Evaluation Metrics ---")
    print(f"Average Final IoU: {avg_iou:.4f}")
    print(f"Average Steps Taken: {avg_steps:.2f}")
    print(f"Localization Accuracy (IoU >= 0.3): {acc_03*100:.2f}%")
    print(f"Localization Accuracy (IoU >= 0.5): {acc_05*100:.2f}%")
    print(f"Localization Accuracy (IoU >= 0.7): {acc_07*100:.2f}%")
    
    if logging:
        os.makedirs('logs', exist_ok=True)
        csv_path = os.path.join('logs', output_file)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Image_ID', 'Ground_Truth', 'Prediction', 'Steps', 'IoU'])
            writer.writeheader()
            writer.writerows(log_data)
        print(f"-> Detailed metrics logged to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Testing (v2)")
    parser.add_argument('--method', type=str, choices=['surrogate', 'ats', 'stdp'], required=True)
    parser.add_argument('--backbone', type=str, choices=['conv', 'vgg16'], default='conv')
    parser.add_argument('--target', type=str, default='mixing')
    parser.add_argument('--num-samples', type=int, default=10) # Test on 10 samples by default
    parser.add_argument('--simulate', type=int, default=10)
    parser.add_argument('--render', action='store_true', help="Show images with bounding boxes")
    parser.add_argument('--logging', action='store_true', help="Log metrics to CSV")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    voc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'VOC2012')
    dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_samples)
    
    if args.method == 'surrogate':
        model = SQNSurrogate(simulation_time=args.simulate, use_vgg16=(args.backbone == 'vgg16'))
    elif args.method == 'ats':
        model = SQNConverted(simulation_time=args.simulate, use_vgg16=(args.backbone == 'vgg16'))
        model.is_snn = True # Set to SNN mode for evaluation
    elif args.method == 'stdp':
        if args.backbone == 'vgg16':
            raise ValueError("STDP method requires raw image input and cannot be used with a VGG16 backbone.")
        model = SQNSTDP()
        model.set_pretrain_mode(False) # Ensure RL head is active
        
    model = model.to(device)
    
    # Load weights
    weight_path = f"weights/{args.method}_{args.target}.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: Weights not found at {weight_path}. Evaluating with random weights.")
        
    # Agent wrapper (optimizer not needed for eval)
    agent = LocalizationAgent(model=model, device=device)
    
    csv_file = f"test_{args.method}_{args.target}_{args.backbone}.csv"
    test_model(agent, dataset, render=args.render, logging=args.logging, output_file=csv_file)

if __name__ == '__main__':
    main()
