import os
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim

def get_optimizer(model, opt_name, lr, weight_decay=0.0):
    """Factory function to create the requested optimizer"""
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    if opt_name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay if weight_decay > 0 else 0.01)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, alpha=0.99, eps=1e-8, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == 'radam':
        return optim.RAdam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def plot_training_results(losses, epsilons, method, target):
    """Save training metrics to CSV and plot them"""
    os.makedirs('logs', exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame({
        'epoch': range(1, len(losses) + 1),
        'loss': losses,
        'epsilon': epsilons
    })
    csv_path = f"logs/{method}_{target}_training_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training logs saved to {csv_path}")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, len(losses) + 1), losses, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(range(1, len(epsilons) + 1), epsilons, color=color, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Training Metrics ({method} - {target})")
    fig.tight_layout()
    
    plot_path = f"logs/{method}_{target}_metrics.png"
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.close()
