import torch
import sys
import os

# Add the parent directory to the system path so we can import from federated_learning
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.config.config import *
from federated_learning.models.vae import GradientVAE
from federated_learning.training.training_utils import train_vae

# Create a simple test gradient
def create_test_gradient(dim=1000):
    return torch.randn(dim)

def main():
    print("\n=== Testing VAE Configuration ===")
    print(f"VAE_EPOCHS from config: {VAE_EPOCHS}")
    print(f"VAE_BATCH_SIZE from config: {VAE_BATCH_SIZE}")
    print(f"VAE_LEARNING_RATE from config: {VAE_LEARNING_RATE}")
    print(f"VAE_DEVICE from config: {VAE_DEVICE}")
    
    # Create a small VAE model
    vae = GradientVAE(
        input_dim=1000,
        hidden_dim=64,
        latent_dim=32,
        dropout_rate=0.1,
        projection_dim=512
    )
    
    # Create some test gradients
    gradients = [create_test_gradient() for _ in range(100)]
    
    # Normalize gradients
    normalized_gradients = []
    for grad in gradients:
        norm = torch.norm(grad) + 1e-8
        normalized_gradients.append(grad / norm)
    
    # Train the VAE (this should use the configuration parameters)
    train_vae(vae, normalized_gradients)
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    main() 