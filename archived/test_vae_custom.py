import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import traceback

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

class SimpleVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128):  # Increased dimensions
        super(SimpleVAE, self).__init__()
        print(f"\nInitializing SimpleVAE with:")
        print(f"  input_dim: {input_dim}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  latent_dim: {latent_dim}")
        
        # Enhanced encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Enhanced decoder with batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        print("Model initialized successfully")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=0.1):  # Increased beta for better regularization
        """Enhanced loss function with better balancing"""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss, recon_loss, kld_loss

def create_test_gradient(size=1000, noise_level=0.0):
    """Create a test gradient with optional noise"""
    try:
        base_grad = torch.randn(size)
        if noise_level > 0:
            noise = torch.randn(size) * noise_level
            grad = base_grad + noise
            # Don't normalize here - we want to preserve the noise impact
            return grad
        return base_grad
    except Exception as e:
        print(f"Error in create_test_gradient: {str(e)}")
        traceback.print_exc()
        raise

def test_vae_reconstruction():
    try:
        print("\n=== Testing VAE Reconstruction Error ===")
        
        # Initialize VAE with improved architecture
        input_dim = 1000
        vae = SimpleVAE(
            input_dim=input_dim,
            hidden_dim=256,  # Increased
            latent_dim=128   # Increased
        )
        
        # Create trusted gradients (low noise)
        print("\nGenerating test data...")
        trusted_gradients = [create_test_gradient(input_dim, noise_level=0.01) for _ in range(100)]
        print(f"Created {len(trusted_gradients)} trusted gradients")
        print(f"Gradient shape: {trusted_gradients[0].shape}")
        
        untrusted_gradients = [create_test_gradient(input_dim, noise_level=1.0) for _ in range(100)]
        print(f"Created {len(untrusted_gradients)} untrusted gradients")
        
        # Convert to tensors for batch processing
        print("\nConverting to tensors...")
        trusted_tensor = torch.stack(trusted_gradients)
        print(f"Trusted tensor shape: {trusted_tensor.shape}")
        
        # Train VAE on trusted gradients with improved parameters
        print("\nTraining VAE on trusted gradients...")
        optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4, weight_decay=1e-4)  # Adjusted learning rate and weight decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        batch_size = 32
        
        for epoch in range(50):  # Increased epochs
            vae.train()
            total_loss = 0
            total_recon = 0
            total_kld = 0
            
            # Process in batches
            for i in range(0, len(trusted_tensor), batch_size):
                batch = trusted_tensor[i:i+batch_size]
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = vae(batch)
                loss, recon_loss, kld_loss = vae.loss_function(recon_batch, batch, mu, logvar)
                
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()
            
            avg_loss = total_loss / (len(trusted_tensor) / batch_size)
            avg_recon = total_recon / (len(trusted_tensor) / batch_size)
            avg_kld = total_kld / (len(trusted_tensor) / batch_size)
            
            # Update learning rate
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/50:")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Reconstruction: {avg_recon:.4f}")
                print(f"  KL Divergence: {avg_kld:.4f}")
                print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Test reconstruction error
        print("\nTesting reconstruction error...")
        vae.eval()
        
        def compute_reconstruction_error(gradient):
            with torch.no_grad():
                recon, _, _ = vae(gradient.unsqueeze(0))
                return F.mse_loss(recon, gradient.unsqueeze(0), reduction='mean').item()
        
        # Calculate reconstruction errors
        print("\nCalculating reconstruction errors...")
        trusted_errors = [compute_reconstruction_error(grad) for grad in trusted_gradients]
        untrusted_errors = [compute_reconstruction_error(grad) for grad in untrusted_gradients]
        
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
        
        # Test with different noise levels (more distinct)
        print("\nTesting with different noise levels...")
        noise_levels = [0.01, 0.25, 0.5, 1.0, 2.0, 4.0]  # More distinct noise levels
        errors_by_noise = []
        
        for noise in noise_levels:
            test_gradients = [create_test_gradient(input_dim, noise_level=noise) for _ in range(20)]
            test_errors = [compute_reconstruction_error(grad) for grad in test_gradients]
            mean_error = np.mean(test_errors)
            std_error = np.std(test_errors)
            errors_by_noise.append(mean_error)
            print(f"Noise level {noise:.2f}: mean error = {mean_error:.4f} ± {std_error:.4f}")
        
        # Verify that error increases with noise level
        for i in range(len(noise_levels)-1):
            assert errors_by_noise[i+1] > errors_by_noise[i], \
                f"Error should increase with noise level, but {noise_levels[i+1]} had lower error than {noise_levels[i]}"
        
        print("\nAll tests passed! The VAE successfully distinguishes between trusted and untrusted gradients.")
    
    except Exception as e:
        print(f"\nError in test_vae_reconstruction: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        test_vae_reconstruction()
    except Exception as e:
        print(f"\nMain error: {str(e)}")
        traceback.print_exc() 