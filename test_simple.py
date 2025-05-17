import torch
import torch.nn.functional as F
import copy
import random
import numpy as np
from federated_learning.training.server import Server
from federated_learning.training.aggregation import aggregate_gradients

def test_fix_verification():
    """Verify that our fixes are properly applied"""
    print("\n=== Testing fixes ===")
    
    # Create a server instance
    server = Server()
    print("Server created successfully")
    
    # Create a random gradient
    gradient_dim = 131466  # CNN model size
    gradients = [torch.randn(gradient_dim) for _ in range(3)]
    
    # Test fix for empty client gradients
    print("\nTesting empty gradients fix...")
    try:
        result = server.aggregate_gradients(0, [], None)
        print("Empty gradients handled correctly ✓")
    except Exception as e:
        print(f"Error with empty gradients: {str(e)} ✗")
    
    # Test aggregation methods
    print("\nTesting aggregation methods...")
    methods = ['fedavg', 'fedprox', 'fedbn', 'fedbn_fedprox']
    for method in methods:
        try:
            kwargs = {'weights': torch.ones(len(gradients)) / len(gradients)}
            if 'fedbn' in method:
                kwargs['model'] = server.global_model
                
            result = aggregate_gradients(
                client_gradients=gradients,
                aggregation_method=method,
                **kwargs
            )
            print(f"{method} aggregation successful ✓")
        except Exception as e:
            print(f"Error with {method}: {str(e)} ✗")
    
    print("\nFix verification completed")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("=== Simple Test for Federated Learning System ===")
    test_fix_verification() 