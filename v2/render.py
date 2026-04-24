import argparse
import torch
import os
import numpy as np
import cv2

from v2.data.voc import VOCDataset
from v2.agents.localization_agent import LocalizationAgent
from v2.models.surrogate import SQNSurrogate
from v2.models.ats import SQNConverted
from v2.models.stdp import SQNSTDP

def render_predictions(agent, dataset, num_images=5):
    print(f"\n--- Rendering Visualizations for {num_images} samples ---")
    
    for idx in range(min(num_images, len(dataset))):
        sample = dataset[idx]
        image = sample['image']
        ground_truth = sample['box']
        
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
        vis_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw ground truth (Green)
        cv2.rectangle(vis_img, (int(ground_truth[0]), int(ground_truth[1])), 
                      (int(ground_truth[2]), int(ground_truth[3])), (0, 255, 0), 2)
        
        # Draw intermediate predictions (Blue, thin)
        for m in masks[:-1]:
             cv2.rectangle(vis_img, (int(m[0]), int(m[1])), 
                      (int(m[2]), int(m[3])), (255, 0, 0), 1)

        # Draw final prediction (Red)
        cv2.rectangle(vis_img, (int(final_mask[0]), int(final_mask[1])), 
                      (int(final_mask[2]), int(final_mask[3])), (0, 0, 255), 2)
                      
        print(f"Sample {idx+1}: Displaying result...")
        cv2.imshow(f"Visualization - Sample {idx+1}", vis_img)
        print("Press any key to see the next image...")
        cv2.waitKey(0)
            
    cv2.destroyAllWindows()
    print("--- Visualization Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Active Object Localization Visualization (v2)")
    parser.add_argument('--method', type=str, choices=['surrogate', 'ats', 'stdp'], required=True)
    parser.add_argument('--backbone', type=str, choices=['conv', 'vgg16', 'resnet18'], default='conv')
    parser.add_argument('--target', type=str, default='mixing')
    parser.add_argument('--num-images', type=int, default=5, help="Number of images to render")
    parser.add_argument('--simulate', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    voc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'VOC2012')
    dataset = VOCDataset(root_dir=voc_dir, target_class=args.target, num_samples=args.num_images)
    
    if args.method == 'surrogate':
        model = SQNSurrogate(simulation_time=args.simulate, backbone_name=args.backbone)
    elif args.method == 'ats':
        model = SQNConverted(simulation_time=args.simulate, backbone_name=args.backbone)
        model.is_snn = True
    elif args.method == 'stdp':
        if args.backbone == 'vgg16':
            raise ValueError("STDP method requires raw image input and cannot be used with a VGG16 backbone.")
        model = SQNSTDP()
        model.set_pretrain_mode(False)
        
    model = model.to(device)
    
    weight_path = f"weights/{args.method}_{args.target}.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Error: Weights not found at {weight_path}. Cannot render without trained weights.")
        return
        
    agent = LocalizationAgent(model=model, device=device)
    render_predictions(agent, dataset, num_images=args.num_images)

if __name__ == '__main__':
    main()
