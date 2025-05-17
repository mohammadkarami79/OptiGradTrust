import torch
import numpy as np
from federated_learning.training.server import Server
from federated_learning.config.config import *

class TestServer(Server):
    """Subclass to override VAE_EPOCHS for faster testing"""
    def _train_models(self):
        """
        Train the VAE and Dual Attention models on root gradients.
        (Override to use fewer epochs)
        """
        print("\n=== Training VAE and Dual Attention Models ===")
        
        # Prepare root gradients
        print("\nPreparing root gradients...")
        
        if len(self.root_gradients) == 0:
            print("Warning: No root gradients available for training")
            return
            
        print(f"Training VAE on {len(self.root_gradients)} normalized gradients")
        
        # Re-train the VAE on root gradients (fewer epochs for testing)
        self.train_vae(self.root_gradients, vae_epochs=2)  # Reduced from VAE_EPOCHS
        
        # Generate training features for dual attention
        print("\n=== Generating Training Data for Dual Attention ===")
        
        # Generate features for honest clients (using root gradients)
        honest_features = self._compute_all_gradient_features(self.root_gradients)
        
        # Generate synthetic malicious features for training
        num_synthetic = max(1, len(honest_features))
        print(f"Generating {num_synthetic} synthetic malicious gradient features...")
        
        malicious_features = self._generate_malicious_features(honest_features)
        
        # Rest of method unchanged...
        feature_names = ["RE", "Root Sim", "Client Sim", "Norm", "Consistency"]
        if honest_features.shape[1] > 5:
            feature_names.append("Shapley")
            
        print("\nTest successful: Generated both honest and malicious features")
        print(f"Honest features shape: {honest_features.shape}")
        print(f"Malicious features shape: {malicious_features.shape}")

def test_full_train_models():
    print("Testing full _train_models method...")
    server = TestServer()  # Use our testing subclass
    
    # Create a dummy root gradient
    dummy_gradient = torch.randn(GRADIENT_DIMENSION)
    server.root_gradients = [dummy_gradient]
    
    # Try to run the _train_models method
    try:
        print("\nCalling _train_models method...")
        server._train_models()
        print("SUCCESS: _train_models completed successfully")
    except Exception as e:
        print(f"ERROR in _train_models: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed")

if __name__ == "__main__":
    test_full_train_models() 