import os
import csv
import torch
import numpy as np

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
