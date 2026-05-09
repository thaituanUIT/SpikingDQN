import os
import torch
import numpy as np

def train_stdp_pretraining(model, dataset, device, stdp_epochs=3, lr_decay=0.5):
    """
    Unsupervised STDP Pre-training phase for the Backbone.
    Runs multiple epochs with decaying STDP learning rates to allow
    the convolutional filters to converge to stable feature detectors.
    After pretraining, validates that the backbone produces meaningful features.
    """
    print(f"\n--- Starting Unsupervised STDP Pre-training ({stdp_epochs} epochs) ---")
    model.set_pretrain_mode(True)
    
    # Store original LRs so we can decay them
    stdp_layers = [model.conv1, model.conv2, model.conv3]
    original_lrs = [(l.lr_plus, l.lr_minus) for l in stdp_layers]
    
    for ep in range(1, stdp_epochs + 1):
        # Decay LR each epoch
        decay_factor = lr_decay ** (ep - 1)
        for layer, (lr_p, lr_m) in zip(stdp_layers, original_lrs):
            layer.lr_plus = lr_p * decay_factor
            layer.lr_minus = lr_m * decay_factor
        
        print(f"STDP Epoch {ep}/{stdp_epochs} (LR decay factor: {decay_factor:.3f})")
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            img = sample['image']
            
            # Format image for STDP
            img_transposed = np.transpose(img, (2, 0, 1))
            img_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float().to(device) / 255.0
            
            # Forward pass triggers STDP weight updates internally
            model(img_tensor, None)
            
            if (idx + 1) % 10 == 0:
                stats = model.get_backbone_stats()
                conv1_fr = stats['conv1']['firing_rate_mean']
                conv2_fr = stats['conv2']['firing_rate_mean']
                conv3_fr = stats['conv3']['firing_rate_mean']
                print(f"[{idx+1}/{len(dataset)}] Firing rates: "
                      f"C1={conv1_fr:.4f} C2={conv2_fr:.4f} C3={conv3_fr:.4f} | "
                      f"Thresh: C1={stats['conv1']['threshold_mean']:.1f} "
                      f"C2={stats['conv2']['threshold_mean']:.1f} "
                      f"C3={stats['conv3']['threshold_mean']:.1f}")
    
    # --- Feature Validation Checkpoint ---
    print("\n  Validating backbone features...")
    model.set_pretrain_mode(False)  # Temporarily disable STDP for clean forward
    
    with torch.no_grad():
        # Run a few images through and check feature statistics
        feature_stats = []
        num_check = min(5, len(dataset))
        for idx in range(num_check):
            sample = dataset[idx]
            img = sample['image']
            img_transposed = np.transpose(img, (2, 0, 1))
            img_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float().to(device) / 255.0
            
            # Get features from backbone
            x = model.dog(img_tensor)
            latencies = model._encode_latencies(x)
            c1 = model.conv1(latencies)
            c1 = model.pool(c1)
            c2 = model.conv2(c1)
            c2 = model.pool(c2)
            c3 = model.conv3(c2)
            c3 = model.pool(c3)
            
            flat = c3.reshape(1, -1)
            nonzero_frac = (flat != 0).float().mean().item()
            feat_std = flat.std().item()
            feat_mean = flat.mean().item()
            feature_stats.append((nonzero_frac, feat_std, feat_mean))
        
        avg_nz = np.mean([s[0] for s in feature_stats])
        avg_std = np.mean([s[1] for s in feature_stats])
        avg_mean = np.mean([s[2] for s in feature_stats])
        
        print(f"\nFeature stats (avg over {num_check} images):")
        print(f"Non-zero fraction: {avg_nz:.4f} | Std: {avg_std:.4f} | Mean: {avg_mean:.4f}")
    
    # Final backbone stats
    final_stats = model.get_backbone_stats()
    print("\nFinal backbone diagnostics:")
    for name, s in final_stats.items():
        print(f"{name}: thresh={s['threshold_mean']:.2f}±{s['threshold_std']:.2f}, "
              f"fire_rate={s['firing_rate_mean']:.4f}±{s['firing_rate_std']:.4f}, "
              f"w_mean={s['weight_mean']:.4f}, w_std={s['weight_std']:.4f}")
    
    print("--- STDP Pre-training Complete ---\n")

from torch.utils.data import WeightedRandomSampler
from helpers.tester import test_model

def run_rl_training(agent, dataset, epochs, epsilon_start=1.0, epsilon_min=0.1, decay_steps=10, early_stop_patience=0, save_mode="none", save_path="weights/best_model.pth", batch_size=20, target_update=1, val_dataset=None, validation_mode='none'):
    """Standard DQN Training Loop"""
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_min) / decay_steps
    
    # Track logs
    history_loss = []
    history_epsilon = []
    
    best_metric = -float('inf') if validation_mode == 'iou' else float('inf')
    patience_counter = 0
    
    # Ensure weights directory exists if saving
    if save_mode != "none":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    weights = dataset.get_sample_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        epoch_loss = []
        epoch_reward = 0
        
        for step_idx, idx in enumerate(sampler):
            sample = dataset[idx]
            image = sample['image']
            ground_truth = sample['box']
            
            history = [-1] * agent.history_size
            height, width, _ = image.shape
            current_mask = np.asarray([0, 0, width, height])
            
            step = 0
            done = False
            img_reward = 0
            
            while not done:
                current_mask, reward, done, history = agent.step(
                    image, history, current_mask, ground_truth, step, epsilon
                )
                
                loss = agent.train_step(batch_size=batch_size)
                if loss > 0:
                    epoch_loss.append(loss)
                    
                img_reward += reward
                step += 1
                
            epoch_reward += img_reward
            
            if (step_idx + 1) % 10 == 0:
                print(f"Image {step_idx+1}: Reward = {img_reward}, Steps = {step}")
        
        avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0
        print(f"Epoch {epoch} Results: Avg Loss = {avg_loss:.4f}, Total Reward = {epoch_reward}, Epsilon = {epsilon:.2f}")
        
        history_loss.append(avg_loss)
        history_epsilon.append(epsilon)
        
        # Update target network
        if epoch % target_update == 0:
            agent.update_target_network()
            print("Target network updated.")
            
        val_iou = None
        val_loss = None
        if validation_mode != 'none' and val_dataset is not None:
            val_iou, val_loss = test_model(agent, val_dataset, verbose=False)
            print(f"Validation: IoU = {val_iou:.4f}, Loss = {val_loss:.4f}")
            
        # Determine tracking metric for early stop and saving
        is_best = False
        if validation_mode == 'iou' and val_dataset is not None:
            if val_iou > best_metric:
                is_best = True
                best_metric = val_iou
        elif validation_mode == 'loss' and val_dataset is not None:
            if val_loss < best_metric:
                is_best = True
                best_metric = val_loss
        else: # 'none' or missing val_dataset
            if avg_loss < best_metric:
                is_best = True
                best_metric = avg_loss
            
        # Save model based on mode
        if save_mode == "best" and is_best:
            torch.save(agent.model.state_dict(), save_path)
            metric_str = f"Val IoU: {best_metric:.4f}" if validation_mode == 'iou' else (f"Val Loss: {best_metric:.4f}" if validation_mode == 'loss' else f"Avg Loss: {best_metric:.4f}")
            print(f"New best model saved with {metric_str}")
        elif save_mode == "epoch":
            epoch_save_path = save_path.replace(".pth", f"_epoch_{epoch}.pth")
            torch.save(agent.model.state_dict(), epoch_save_path)
            print(f"Model saved for epoch {epoch} to {epoch_save_path}")

        if is_best:
            patience_counter = 0
        else:
            patience_counter += 1

        if epsilon > epsilon_min:
            epsilon -= epsilon_decay
            
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            metric_name = "Avg Loss" if validation_mode == 'none' else f"Val {validation_mode.upper()}"
            print(f"Early stopping triggered at epoch {epoch}. No improvement in {metric_name} for {early_stop_patience} epochs.")
            break

    return history_loss, history_epsilon
