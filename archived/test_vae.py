import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.models.vae import GradientVAE

def create_test_gradient(size=1000, noise_level=0.0):
    """Create a test gradient with optional noise"""
    base_grad = torch.randn(size)
    if noise_level > 0:
        noise = torch.randn(size) * noise_level
        return base_grad + noise
    return base_grad

def test_vae_reconstruction():
    print("\n=== Testing VAE Reconstruction Error ===")
    
    # Initialize VAE with more complex configuration
    input_dim = 1000
    vae = GradientVAE(
        input_dim=input_dim,
        hidden_dim=256,  # Increased hidden dimension
        latent_dim=128,  # Increased latent dimension
        dropout_rate=0.2,  # Increased dropout
        projection_dim=None  # Disable projection for testing
    )
    
    # Create trusted gradients (low noise)
    trusted_gradients = [create_test_gradient(input_dim, noise_level=0.01) for _ in range(100)]  # More samples
    
    # Create untrusted gradients (high noise)
    untrusted_gradients = [create_test_gradient(input_dim, noise_level=0.5) for _ in range(100)]  # More samples
    
    # Normalize all gradients
    def normalize_gradients(gradients):
        normalized = []
        for grad in gradients:
            norm = torch.norm(grad) + 1e-8
            normalized.append(grad / norm)
        return normalized
    
    trusted_gradients = normalize_gradients(trusted_gradients)
    untrusted_gradients = normalize_gradients(untrusted_gradients)
    
    # Train VAE on trusted gradients
    print("\nTraining VAE on trusted gradients...")
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    
    # Increase training epochs
    for epoch in range(30):  # More epochs
        epoch_loss = 0
        for grad in trusted_gradients:
            optimizer.zero_grad()
            recon, mu, logvar = vae(grad.unsqueeze(0))
            loss = vae.loss_function(recon, grad.unsqueeze(0), mu, logvar)[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/30, Loss: {epoch_loss/len(trusted_gradients):.4f}")
    
    # Test reconstruction error
    print("\nTesting reconstruction error...")
    vae.eval()
    
    # Calculate reconstruction errors
    trusted_errors = []
    untrusted_errors = []
    
    with torch.no_grad():
        for grad in trusted_gradients:
            # Calculate reconstruction error directly
            recon, _, _ = vae(grad.unsqueeze(0))
            error = F.mse_loss(recon, grad.unsqueeze(0), reduction='mean').item()
            trusted_errors.append(error)
            
        for grad in untrusted_gradients:
            # Calculate reconstruction error directly
            recon, _, _ = vae(grad.unsqueeze(0))
            error = F.mse_loss(recon, grad.unsqueeze(0), reduction='mean').item()
            untrusted_errors.append(error)
    
    # Calculate statistics
    trusted_mean = np.mean(trusted_errors)
    trusted_std = np.std(trusted_errors)
    untrusted_mean = np.mean(untrusted_errors)
    untrusted_std = np.std(untrusted_errors)
    
    print("\nReconstruction Error Statistics:")
    print(f"Trusted gradients: mean={trusted_mean:.4f}, std={trusted_std:.4f}")
    print(f"Untrusted gradients: mean={untrusted_mean:.4f}, std={untrusted_std:.4f}")
    
    # Verify that untrusted gradients have higher reconstruction error
    assert untrusted_mean > trusted_mean, "Untrusted gradients should have higher reconstruction error"
    
    # Calculate separation score (how well the VAE distinguishes between trusted and untrusted)
    separation_score = (untrusted_mean - trusted_mean) / (trusted_std + untrusted_std)
    print(f"\nSeparation score: {separation_score:.4f}")
    print("(Higher score indicates better separation between trusted and untrusted gradients)")
    
    # Test with different noise levels
    print("\nTesting with different noise levels...")
    noise_levels = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    errors_by_noise = []
    
    for noise in noise_levels:
        test_gradients = [create_test_gradient(input_dim, noise_level=noise) for _ in range(20)]  # More samples
        test_gradients = normalize_gradients(test_gradients)
        
        with torch.no_grad():
            errors = []
            for grad in test_gradients:
                recon, _, _ = vae(grad.unsqueeze(0))
                error = F.mse_loss(recon, grad.unsqueeze(0), reduction='mean').item()
                errors.append(error)
            mean_error = np.mean(errors)
            errors_by_noise.append(mean_error)
            print(f"Noise level {noise:.2f}: mean error = {mean_error:.4f}")
    
    # Verify that error increases with noise level
    for i in range(len(noise_levels)-1):
        assert errors_by_noise[i+1] > errors_by_noise[i], \
            f"Error should increase with noise level, but {noise_levels[i+1]} had lower error than {noise_levels[i]}"
    
    print("\nAll tests passed! The VAE successfully distinguishes between trusted and untrusted gradients.")

if __name__ == "__main__":
    test_vae_reconstruction() 