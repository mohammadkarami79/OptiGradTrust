import torch
import os
import sys
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing global model update with FedBN...")
    from federated_learning.config.config import *
    from federated_learning.training.server import Server
    from federated_learning.utils.model_utils import update_model_with_gradient
    
    # Set aggregation method to 'fedavg' temporarily to test complete update
    import federated_learning.config.config as config
    original_method = config.AGGREGATION_METHOD
    print(f"Original aggregation method: {original_method}")
    
    # Temporarily switch to fedavg for testing
    config.AGGREGATION_METHOD = 'fedavg'
    print(f"Temporarily set aggregation method to: {config.AGGREGATION_METHOD}")
    
    # Initialize server
    print("\n=== Initializing Server ===")
    server = Server()
    
    # Create a test gradient
    print("\n=== Creating Test Gradient ===")
    model = server.global_model
    device = next(model.parameters()).device
    gradient_size = sum(p.numel() for p in model.parameters())
    print(f"Model has {gradient_size} parameters")
    
    # Create a gradient with random values
    gradient = torch.randn(gradient_size, device=device)
    print(f"Created gradient with shape {gradient.shape} on {gradient.device}")
    print(f"Gradient norm: {torch.norm(gradient).item():.4f}")
    
    # Store initial parameter values for comparison
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.clone().detach()
        # Print first value of each parameter
        flat_param = param.flatten()
        if flat_param.numel() > 0:
            print(f"Parameter '{name}': shape {param.shape}, first value {flat_param[0].item():.6f}")
    
    print(f"\nTotal parameter count: {sum(p.numel() for p in model.parameters())}")
    
    # Directly update model with server._update_global_model
    print("\n=== Updating Model with server._update_global_model ===")
    server._update_global_model(gradient)
    
    # Check for parameter changes
    print("\n=== Checking Parameter Changes ===")
    changed_params = []
    unchanged_params = []
    
    for name, param in model.named_parameters():
        if name in initial_params:
            # Calculate the absolute sum of differences
            param_diff = torch.sum(torch.abs(param - initial_params[name])).item()
            flat_param = param.flatten()
            initial_flat = initial_params[name].flatten()
            
            if param_diff > 1e-6:
                changed_params.append((name, param_diff))
                print(f"✅ Parameter '{name}' changed: {param_diff:.6f}")
                print(f"   - Before: {initial_flat[0].item():.6f}, After: {flat_param[0].item():.6f}")
            else:
                unchanged_params.append(name)
                print(f"❌ Parameter '{name}' unchanged: {param_diff:.6f}")
                print(f"   - Before: {initial_flat[0].item():.6f}, After: {flat_param[0].item():.6f}")
    
    # Print summary
    print("\n=== Update Summary ===")
    print(f"Total parameters: {len(initial_params)}")
    print(f"Changed parameters: {len(changed_params)}")
    print(f"Unchanged parameters: {len(unchanged_params)}")
    
    if changed_params:
        print("\nChanged parameters:")
        for name, diff in changed_params[:5]:  # Show top 5
            print(f" - {name}: {diff:.6f}")
        if len(changed_params) > 5:
            print(f" - ... and {len(changed_params) - 5} more")
    
    if unchanged_params:
        print("\nUnchanged parameters (might include BatchNorm in FedBN mode):")
        for name in unchanged_params[:5]:  # Show top 5
            print(f" - {name}")
        if len(unchanged_params) > 5:
            print(f" - ... and {len(unchanged_params) - 5} more")
    
    # Restore original aggregation method
    config.AGGREGATION_METHOD = original_method
    print(f"\nRestored aggregation method to: {config.AGGREGATION_METHOD}")
    
    print("\n=== Test Completed Successfully ===")
    
except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc() 