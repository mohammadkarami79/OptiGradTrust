"""
Test script to verify that all aggregation methods are working correctly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from federated_learning.models.cnn import CNNMnist
from federated_learning.training.aggregation import aggregate_gradients

# Create a simple CNN model for testing
def create_test_model():
    model = CNNMnist()
    # Add batch normalization to test FedBN
    model.fc1_bn = nn.BatchNorm1d(320)
    return model

def main():
    print("=== Testing All Aggregation Methods ===")
    
    # Create a test model
    model = create_test_model()
    print(f"Created test model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy client gradients
    num_clients = 3
    gradients = []
    for i in range(num_clients):
        # Create a flattened gradient vector with some pattern
        grad = torch.zeros(sum(p.numel() for p in model.parameters() if p.requires_grad))
        # Add some values based on client index to make them different
        grad += torch.randn_like(grad) * (i + 1) * 0.1
        gradients.append(grad)
    
    print(f"Created {len(gradients)} client gradients with shape {gradients[0].shape}")
    
    # Test each aggregation method
    aggregation_methods = ['fedavg', 'fedprox', 'fedadmm', 'fedbn', 'feddwa', 'fednova', 'fedbn_fedprox']
    
    for method in aggregation_methods:
        print(f"\n--- Testing {method} ---")
        try:
            # Set up any method-specific arguments
            kwargs = {}
            
            if method == 'fedbn' or method == 'fedbn_fedprox':
                kwargs['model'] = model
            
            if method == 'feddwa':
                kwargs['client_metrics'] = [0.8, 0.9, 0.7]  # Example metrics
                kwargs['weighting_method'] = 'accuracy'
            
            if method == 'fedadmm':
                kwargs['rho'] = 0.1
                kwargs['sigma'] = 0.1
                kwargs['iterations'] = 2
            
            if method == 'fednova':
                kwargs['client_steps'] = [10, 12, 8]  # Example step counts
            
            # Aggregate gradients with the current method
            result = aggregate_gradients(gradients, aggregation_method=method, **kwargs)
            
            # Print basic statistics about the result
            print(f"Result shape: {result.shape}")
            print(f"Result norm: {torch.norm(result).item():.4f}")
            
            # For FedBN, check if it's correctly handling batch normalization parameters
            if method in ['fedbn', 'fedbn_fedprox']:
                # Count number of parameters in a batch normalization layer
                bn_params = sum(p.numel() for name, p in model.named_parameters() 
                               if 'bn' in name or 'downsample.1' in name)
                if bn_params > 0:
                    print(f"Model has {bn_params} batch normalization parameters")
                else:
                    print("Warning: No batch normalization parameters found in model")
            
        except Exception as e:
            print(f"Error testing {method}: {str(e)}")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main() 