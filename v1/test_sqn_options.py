import torch
from RLSNN.compact.SQN import SQN

def test_sqn_options():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test different combinations
    configs = [
        {'encoding': 'constant', 'decoding': 'potential'},
        {'encoding': 'poisson', 'decoding': 'potential'},
        {'encoding': 'constant', 'decoding': 'spikes'},
        {'encoding': 'equidistant', 'decoding': 'potential'}
    ]
    
    for config in configs:
        print(f"\nTesting config: {config}")
        model = SQN(input_dim=(3, 224, 224), output_dim=9, history_dim=90, **config).to(device)
        
        image_tensor = torch.abs(torch.randn(2, 3, 224, 224)).to(device) # Abs for poisson probs
        history_tensor = torch.randn(2, 90).to(device)
        
        outputs = model(image_tensor, history_tensor)
        print(f"Output shape: {outputs.shape}")
        assert outputs.shape == (2, 9)
        
        # Check for NaN
        if torch.isnan(outputs).any():
            print("WARNING: NaN detected in outputs!")
        else:
            print("Outputs are valid.")

    print("\nAll SQN Option Tests Passed!")

if __name__ == "__main__":
    test_sqn_options()
