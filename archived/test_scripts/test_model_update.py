import torch
import torch.nn.functional as F
import numpy as np
import copy
from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.training.server import Server

def test_model_update():
    """
    Test that the global model updates correctly when applying gradients.
    """
    print("\n=== Testing Global Model Update ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create a server instance
    server = Server()
    
    # Get initial model parameters
    initial_params = {}
    for name, param in server.global_model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
    
    # Create a random gradient
    gradient_size = sum(p.numel() for p in server.global_model.parameters() if p.requires_grad)
    random_gradient = torch.randn(gradient_size, device=server.device)
    
    # Update global model
    print("\nUpdating global model with random gradient...")
    updated_model = server._update_global_model(random_gradient, 0)
    server.global_model = updated_model
    
    # Check if parameters have changed
    params_changed = False
    total_param_diff = 0.0
    for name, param in server.global_model.named_parameters():
        if param.requires_grad and name in initial_params:
            param_diff = torch.norm(param.data - initial_params[name]).item()
            total_param_diff += param_diff
            if param_diff > 0:
                params_changed = True
                print(f"Parameter {name} changed by {param_diff:.6f}")
    
    print(f"Total parameter difference: {total_param_diff:.6f}")
    print(f"Parameters changed: {params_changed}")
    
    return params_changed and total_param_diff > 0

if __name__ == "__main__":
    success = test_model_update()
    print(f"\nTest {'passed' if success else 'failed'}") 