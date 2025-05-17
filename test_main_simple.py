import torch
import numpy as np
import random
import os
import sys
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    from federated_learning.config.config import *
    from federated_learning.training.server import Server
    
    # Set random seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
    
    print("\n=== Initializing Server ===")
    server = Server()
    
    print("\nServer initialized successfully!")
    print(f"Server has global model: {server.global_model is not None}")
    print(f"Server has VAE: {server.vae is not None}")
    print(f"Server has dual attention: {server.dual_attention is not None}")
    
    # Check model update functionality
    print("\n=== Testing Model Update ===")
    from federated_learning.utils.model_utils import update_model_with_gradient
    
    # Create a simple gradient
    device = next(server.global_model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Create a small gradient
    gradient = torch.ones(1000, device=device)
    print(f"Created gradient with shape {gradient.shape} on {gradient.device}")
    
    # Get a copy of the first parameter before update
    first_param = next(server.global_model.parameters())
    before_value = first_param[0][0].item() if len(first_param.shape) > 1 else first_param[0].item()
    print(f"Parameter before update: {before_value:.6f}")
    
    # Update model
    learning_rate = 0.01
    print(f"Applying update with learning rate: {learning_rate}")
    
    # Create a copy of the model
    test_model = type(server.global_model)().to(device)
    test_model.load_state_dict(server.global_model.state_dict())
    
    # Update the model
    updated_model, total_change, avg_change = update_model_with_gradient(
        test_model, gradient, learning_rate
    )
    
    # Get first parameter after update
    first_param = next(updated_model.parameters())
    after_value = first_param[0][0].item() if len(first_param.shape) > 1 else first_param[0].item()
    print(f"Parameter after update: {after_value:.6f}")
    print(f"Parameter change: {after_value - before_value:.6f}")
    print(f"Total change: {total_change:.6f}")
    print(f"Average change: {avg_change:.6f}")
    
    print("\n=== Test Completed Successfully ===")
    
except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc() 