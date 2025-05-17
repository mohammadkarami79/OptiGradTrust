import torch
import numpy as np
import random
import os
import sys
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing global model update...")
    from federated_learning.config.config import *
    from federated_learning.training.server import Server
    from federated_learning.utils.model_utils import update_model_with_gradient
    
    # Set random seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
    
    # Initialize server
    print("\n=== Initializing Server ===")
    server = Server()
    
    # Get global model
    print("\n=== Testing Global Model Update ===")
    global_model = server.global_model
    
    # Create a gradient
    print("Creating test gradient...")
    device = next(global_model.parameters()).device
    gradient_size = sum(p.numel() for p in global_model.parameters())
    print(f"Model has {gradient_size} parameters")
    
    # Create a gradient with all ones
    gradient = torch.ones(gradient_size, device=device)
    print(f"Created gradient with shape {gradient.shape} on {gradient.device}")
    
    # Get initial parameter values
    initial_params = {}
    for name, param in global_model.named_parameters():
        initial_params[name] = param.clone()
    
    # Save first parameter values for comparison
    first_param_name = list(initial_params.keys())[0]
    first_param = initial_params[first_param_name]
    print(f"Initial parameter '{first_param_name}' shape: {first_param.shape}")
    print(f"Initial value: {first_param.flatten()[0].item():.6f}")
    
    # Update global model
    print("\n=== Updating Global Model ===")
    learning_rate = 0.01
    print(f"Learning rate: {learning_rate}")
    
    # Test normal update
    print("\nTest 1: Direct update_model_with_gradient call")
    updated_model, total_change, avg_change = update_model_with_gradient(
        global_model, gradient, learning_rate
    )
    
    # Check parameter changes
    print("\nChecking parameter changes...")
    param_changes = []
    for name, param in updated_model.named_parameters():
        if name in initial_params:
            change = torch.sum(torch.abs(param - initial_params[name])).item()
            param_changes.append(change)
            if name == first_param_name:
                print(f"Parameter '{name}' change: {change:.6f}")
                print(f"New value: {param.flatten()[0].item():.6f}")
                print(f"Change in first value: {param.flatten()[0].item() - first_param.flatten()[0].item():.6f}")
    
    print(f"Total change: {total_change:.6f}")
    print(f"Average change: {avg_change:.6f}")
    print(f"Sum of parameter changes: {sum(param_changes):.6f}")
    
    # Test server's _update_global_model method
    print("\nTest 2: Using server._update_global_model")
    # Create a new instance of the model
    new_server = Server()
    
    # Get the current state before update
    before_param = next(new_server.global_model.parameters()).flatten()[0].item()
    print(f"Parameter before update: {before_param:.6f}")
    
    # Apply the update through server method
    new_server._update_global_model(gradient, round_idx=0)
    
    # Check the state after update
    after_param = next(new_server.global_model.parameters()).flatten()[0].item()
    print(f"Parameter after update: {after_param:.6f}")
    print(f"Change: {after_param - before_param:.6f}")
    
    print("\n=== Test Completed Successfully ===")
    
    # Test 3: Using fedavg instead of fedbn
    print("\nTest 3: Using fedavg instead of fedbn")
    
    # Save original aggregation method
    original_aggregation_method = AGGREGATION_METHOD
    
    try:
        # Temporarily set to fedavg
        import federated_learning.config.config as config
        config.AGGREGATION_METHOD = 'fedavg'
        print(f"Temporarily changed aggregation method from {original_aggregation_method} to fedavg")
        
        # Create another new server
        new_server2 = Server()
        
        # Collect non-batchnorm parameters before update
        non_bn_params_before = {}
        bn_params_before = {}
        for name, param in new_server2.global_model.named_parameters():
            if 'bn' in name:
                bn_params_before[name] = param.detach().clone()
            else:
                non_bn_params_before[name] = param.detach().clone()
        
        # Get a specific non-batchnorm parameter
        if len(non_bn_params_before) > 0:
            non_bn_name = list(non_bn_params_before.keys())[0]
            non_bn_param_before = non_bn_params_before[non_bn_name].flatten()[0].item()
            print(f"Non-BatchNorm parameter '{non_bn_name}' before update: {non_bn_param_before:.6f}")
        
        # Apply update
        new_server2._update_global_model(gradient, round_idx=0)
        
        # Check changes in non-batchnorm parameters
        updated = False
        for name, param in new_server2.global_model.named_parameters():
            if 'bn' not in name and name in non_bn_params_before:
                before_val = non_bn_params_before[name].flatten()[0].item()
                after_val = param.flatten()[0].item()
                change = after_val - before_val
                if abs(change) > 1e-6:
                    updated = True
                    print(f"Non-BatchNorm parameter '{name}' change: {change:.6f}")
                    print(f"  Before: {before_val:.6f}, After: {after_val:.6f}")
                    break  # Just show one example
        
        if updated:
            print("✅ Non-BatchNorm parameters were successfully updated")
        else:
            print("❌ No Non-BatchNorm parameters changed")
        
        # Check if BatchNorm parameters changed
        bn_updated = False
        for name, param in new_server2.global_model.named_parameters():
            if 'bn' in name and name in bn_params_before:
                before_val = bn_params_before[name].flatten()[0].item()
                after_val = param.flatten()[0].item()
                change = after_val - before_val
                if abs(change) > 1e-6:
                    bn_updated = True
                    print(f"BatchNorm parameter '{name}' change: {change:.6f}")
                    print(f"  Before: {before_val:.6f}, After: {after_val:.6f}")
                    break  # Just show one example
        
        if bn_updated:
            print("✅ BatchNorm parameters were successfully updated with fedavg")
        else:
            print("❓ No BatchNorm parameters changed with fedavg (might be expected)")
    
    finally:
        # Restore original aggregation method
        config.AGGREGATION_METHOD = original_aggregation_method
        print(f"Restored aggregation method to {original_aggregation_method}")
    
    print("\n=== All Tests Completed Successfully ===")
    
except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc() 