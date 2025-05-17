import torch
import numpy as np
from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.training.server import Server

def test_simple_update():
    """
    Simple test to verify that the global model updates correctly when applying gradients.
    """
    print("\n=== Testing Simple Model Update ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create a server instance
    server = Server()
    
    # Get initial model parameters for a specific layer
    initial_weight = server.global_model.conv1.weight.data.clone()
    initial_bias = server.global_model.conv1.bias.data.clone()
    
    # Create a simple gradient that would increase all weights by 0.1
    gradient_size = sum(p.numel() for p in server.global_model.parameters() if p.requires_grad)
    gradient = torch.ones(gradient_size, device=server.device) * 0.1
    
    # Update global model
    print("\nUpdating global model with constant gradient...")
    updated_model = server._update_global_model(gradient, 0)
    server.global_model = updated_model
    
    # Check if parameters have changed
    weight_diff = torch.norm(server.global_model.conv1.weight.data - initial_weight).item()
    bias_diff = torch.norm(server.global_model.conv1.bias.data - initial_bias).item()
    
    print(f"Weight difference: {weight_diff:.6f}")
    print(f"Bias difference: {bias_diff:.6f}")
    
    # Check if the change is approximately what we expect
    # With learning rate of 0.01, a gradient of 0.1 should change weights by about 0.001 per parameter
    expected_change_per_param = 0.1 * LR
    num_weight_params = server.global_model.conv1.weight.numel()
    expected_weight_change = expected_change_per_param * np.sqrt(num_weight_params)
    
    print(f"Expected change per parameter: {expected_change_per_param:.6f}")
    print(f"Expected weight norm change: {expected_weight_change:.6f}")
    
    # Check if the model was updated correctly
    success = weight_diff > 0 and bias_diff > 0
    
    return success

if __name__ == "__main__":
    success = test_simple_update()
    print(f"\nTest {'passed' if success else 'failed'}") 