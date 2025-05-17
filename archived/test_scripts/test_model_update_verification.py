import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.training.server import Server

def test_model_update_verification():
    """
    Test to verify that the global model is correctly updated with gradients.
    This test checks if:
    1. The model parameters change after applying a gradient
    2. The model parameters change by the expected amount based on the learning rate
    """
    print("\n=== Testing Model Update Verification ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Create a server instance
    print("Creating server instance...")
    server = Server()
    print(f"Server device: {server.device}")
    
    # Get initial model parameters
    initial_params = {}
    param_count = 0
    for name, param in server.global_model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
            param_count += param.numel()
            print(f"Parameter {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    print(f"Total parameters: {param_count}")
    
    # Create a constant gradient (all ones) for testing
    gradient_size = sum(p.numel() for p in server.global_model.parameters() if p.requires_grad)
    print(f"Gradient size: {gradient_size}")
    constant_gradient = torch.ones(gradient_size, device=server.device)
    
    # Expected change per parameter with learning rate LR
    expected_change_per_param = LR
    print(f"Learning rate: {LR}")
    print(f"Expected change per parameter: {expected_change_per_param}")
    
    # Update global model
    print("\nUpdating global model with constant gradient...")
    try:
        updated_model = server._update_global_model(constant_gradient, 0)
        print("Model update completed successfully")
    except Exception as e:
        print(f"Error during model update: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify model was returned correctly
    print(f"Updated model returned: {updated_model is server.global_model}")
    
    # Check if parameters have changed
    total_param_diff = 0.0
    max_param_diff = 0.0
    max_diff_name = ""
    
    for name, param in server.global_model.named_parameters():
        if param.requires_grad and name in initial_params:
            # Calculate difference
            param_diff = torch.mean(torch.abs(param.data - initial_params[name])).item()
            total_param_diff += param_diff
            
            # Track maximum difference
            if param_diff > max_param_diff:
                max_param_diff = param_diff
                max_diff_name = name
            
            # Check if the difference is close to the expected change
            is_close = abs(param_diff - expected_change_per_param) < 1e-5
            print(f"Parameter {name}: diff={param_diff:.6f}, expected={expected_change_per_param:.6f}, close={is_close}")
    
    print(f"\nTotal parameter difference: {total_param_diff:.6f}")
    print(f"Maximum parameter difference: {max_param_diff:.6f} in {max_diff_name}")
    
    # Check if the model was updated correctly
    success = total_param_diff > 0 and abs(max_param_diff - expected_change_per_param) < 1e-5
    
    return success

if __name__ == "__main__":
    success = test_model_update_verification()
    print(f"\nTest {'passed' if success else 'failed'}") 