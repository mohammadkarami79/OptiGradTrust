import torch
import numpy as np
from federated_learning.training.server import Server
from federated_learning.config.config import *

def test_train_models_steps():
    print("Testing steps of _train_models method...")
    server = Server()
    
    # Create a dummy root gradient
    dummy_gradient = torch.randn(GRADIENT_DIMENSION)
    server.root_gradients = [dummy_gradient]
    
    # Test the train_vae method
    try:
        print("\nTesting train_vae method...")
        server.train_vae(server.root_gradients, vae_epochs=2)
        print("SUCCESS: train_vae works")
    except Exception as e:
        print(f"ERROR in train_vae: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test the compute_all_gradient_features method directly
    try:
        print("\nTesting _compute_all_gradient_features method directly...")
        features = server._compute_all_gradient_features(server.root_gradients)
        print("SUCCESS: _compute_all_gradient_features works")
        print(f"Features shape: {features.shape}")
    except Exception as e:
        print(f"ERROR in _compute_all_gradient_features: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test _generate_malicious_features
    try:
        print("\nTesting _generate_malicious_features method...")
        honest_features = server._compute_all_gradient_features(server.root_gradients)
        malicious_features = server._generate_malicious_features(honest_features)
        print("SUCCESS: _generate_malicious_features works")
        print(f"Malicious features shape: {malicious_features.shape}")
    except Exception as e:
        print(f"ERROR in _generate_malicious_features: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed")

if __name__ == "__main__":
    test_train_models_steps() 