import torch
import numpy as np
from federated_learning.training.server import Server
from federated_learning.config.config import *

def test_compute_all_gradient_features():
    print("Testing _compute_all_gradient_features method...")
    server = Server()
    
    # Create a dummy root gradient
    dummy_gradient = torch.randn(GRADIENT_DIMENSION)
    server.root_gradients = [dummy_gradient]
    
    # Test the method
    try:
        features = server._compute_all_gradient_features([dummy_gradient])
        print("SUCCESS: _compute_all_gradient_features works")
        print(f"Features shape: {features.shape}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    print("Test completed")

if __name__ == "__main__":
    test_compute_all_gradient_features() 