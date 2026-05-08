import os
import cv2
import torch
import numpy as np
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
