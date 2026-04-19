import torch
import torch.nn as nn
from SQN import SQN

def test_sqn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize standalone SQN model
    model = SQN(input_dim=(3, 224, 224), output_dim=9, history_dim=90).to(device)
    
    # Create dummy inputs
    image_tensor = torch.randn(2, 3, 224, 224).to(device) # Batch size of 2
    history_tensor = torch.randn(2, 90).to(device)
    
    # Forward pass
    print("Testing forward pass...")
    outputs = model(image_tensor, history_tensor)
    
    print(f"Output shape: {outputs.shape}") # Should be [2, 9]
    assert outputs.shape == (2, 9), f"Expected shape (2, 9), but got {outputs.shape}"
    
    # Gradient check
    print("Testing backward pass...")
    loss = outputs.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient found for {name}")
        else:
            print(f"WARNING: No gradient for {name}")
            
    print("SQN Standalone Test Passed!")

if __name__ == "__main__":
    test_sqn()
