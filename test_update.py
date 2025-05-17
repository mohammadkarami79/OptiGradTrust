import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from federated_learning.utils.model_utils import update_model_with_gradient
    
    print("\n=== Testing Model Update Functionality ===")
    
    # Create a simple model
    model = nn.Linear(10, 5)
    print(f"Model created: {model}")
    
    # Create a gradient
    gradient = torch.ones(55)  # 5*10 (weights) + 5 (bias)
    print(f"Gradient shape: {gradient.shape}, gradient norm: {torch.norm(gradient).item():.4f}")
    
    # Set learning rate
    lr = 0.1
    print(f"Learning rate: {lr}")
    
    # Record initial weights
    initial_weights = model.weight.clone()
    initial_bias = model.bias.clone()
    print(f"Initial weight[0,0]: {initial_weights[0,0].item():.6f}")
    print(f"Initial bias[0]: {initial_bias[0].item():.6f}")
    
    # Update model
    print("\nUpdating model...")
    updated_model, total_change, avg_change = update_model_with_gradient(model, gradient, lr)
    
    # Verify changes
    final_weights = updated_model.weight
    final_bias = updated_model.bias
    print(f"Final weight[0,0]: {final_weights[0,0].item():.6f}")
    print(f"Final bias[0]: {final_bias[0].item():.6f}")
    
    # Calculate actual changes
    weight_diff = torch.sum(torch.abs(final_weights - initial_weights)).item()
    bias_diff = torch.sum(torch.abs(final_bias - initial_bias)).item()
    actual_diff = weight_diff + bias_diff
    
    print(f"\nWeight difference: {weight_diff:.6f}")
    print(f"Bias difference: {bias_diff:.6f}")
    print(f"Total difference: {actual_diff:.6f}")
    print(f"Reported total change: {total_change:.6f}")
    print(f"Reported average change: {avg_change:.6f}")
    
    # Check if changes match expectations
    expected_change = gradient.sum().item() * lr
    print(f"Expected total change: {expected_change:.6f}")
    
    if abs(total_change - actual_diff) < 1e-6:
        print("\n✅ SUCCESS: Total change calculation is correct")
    else:
        print("\n❌ ERROR: Total change calculation is incorrect")
    
    if total_change > 0:
        print("✅ SUCCESS: Model parameters were updated correctly")
    else:
        print("❌ ERROR: Model parameters were not updated")
    
    # Test with device conversion
    print("\n=== Testing Model Update with Device Conversion ===")
    
    # Create a model on CPU and gradient on device
    cpu_model = nn.Linear(10, 5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        gradient_device = gradient.to(device)
        print(f"Model is on CPU, gradient is on {device}")
        
        # Record initial weights
        initial_weights = cpu_model.weight.clone()
        
        # Update model
        print("Updating model with cross-device gradient...")
        try:
            updated_model, total_change, avg_change = update_model_with_gradient(cpu_model, gradient_device, lr)
            print(f"Update successful! Total change: {total_change:.6f}")
            
            if total_change > 0:
                print("✅ SUCCESS: Device conversion is working correctly")
            else:
                print("❌ ERROR: Device conversion failed (zero change)")
        except Exception as e:
            print(f"❌ ERROR: Device conversion failed with exception: {str(e)}")
    else:
        print("Skipping device conversion test (CUDA not available)")
    
    print("\n=== Test Complete ===")
    
except Exception as e:
    print(f"Test failed with error: {str(e)}")
    import traceback
    traceback.print_exc() 