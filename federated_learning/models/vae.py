import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HashProjection:
    """Memory-efficient projection using a hash-based approach"""
    def __init__(self, input_dim, output_dim, seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        # Create multiple hash functions for stable projection
        self.num_hashes = 4
        self.hash_seeds = [seed + i for i in range(self.num_hashes)]
        
    def __call__(self, x):
        # x shape: [batch_size, input_dim]
        device = x.device
        batch_size = x.shape[0]
        result = torch.zeros((batch_size, self.output_dim), device=device)
        
        # Use multiple hash functions and average results
        for seed_idx in range(self.num_hashes):
            # Process in chunks for memory efficiency
            chunk_size = 10000
            for chunk_start in range(0, self.input_dim, chunk_size):
                chunk_end = min(chunk_start + chunk_size, self.input_dim)
                chunk_indices = torch.arange(chunk_start, chunk_end)
                
                # Deterministic hash mapping from input to output indices
                np.random.seed(self.hash_seeds[seed_idx])
                output_indices = torch.tensor(
                    np.random.randint(0, self.output_dim, size=len(chunk_indices)),
                    device=device,
                    dtype=torch.int64  # Explicitly set dtype to int64
                )
                
                # Deterministic hash mapping for signs (+1 or -1)
                np.random.seed(self.hash_seeds[seed_idx] + 1000)
                signs = torch.tensor(
                    np.random.choice([-1.0, 1.0], size=len(chunk_indices)),
                    device=device
                )
                
                # Process the current chunk
                x_chunk = x[:, chunk_indices]
                
                # For each input feature, add its value to the corresponding output feature
                for i, (out_idx, sign) in enumerate(zip(output_indices, signs)):
                    # Ensure out_idx is int64 and has correct shape
                    out_idx_reshaped = out_idx.view(1, 1).expand(batch_size, 1).to(torch.int64)
                    result.scatter_add_(1, out_idx_reshaped, x_chunk[:, i:i+1] * sign)
        
        # Scale the result for proper variance
        result = result / (self.num_hashes * (self.input_dim / self.output_dim) ** 0.5)
        return result

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dims=None):
        super(VAE, self).__init__()
        
        # Default hidden dimensions with more efficient progressive reduction
        if hidden_dims is None:
            hidden_dims = [
                min(input_dim // 8, 16384),
                min(input_dim // 32, 4096),
                min(input_dim // 128, 1024),
                min(input_dim // 512, 256),
            ]
        
        # Memory-efficient projection for very large input dimensions
        if input_dim > 100000:
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )
            self.use_projection = True
        else:
            self.use_projection = False
        
        # Encoder
        modules = []
        in_channels = hidden_dims[0] if self.use_projection else input_dim
        
        # Build encoder with layer normalization instead of batch normalization
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LayerNorm(h_dim),  # Changed from BatchNorm1d to LayerNorm
                    nn.LeakyReLU(),
                    nn.Dropout(0.2)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        modules = []
        hidden_dims.reverse()
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])
        
        # Build decoder with layer normalization instead of batch normalization
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),  # Changed from BatchNorm1d to LayerNorm
                    nn.LeakyReLU(),
                    nn.Dropout(0.2)
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final layer
        if self.use_projection:
            self.final_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[0]),
                nn.Tanh()  # Normalize output
            )
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dims[0], input_dim),
                nn.Tanh()  # Normalize output
            )
        else:
            self.final_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], input_dim),
                nn.Tanh()  # Normalize output
            )
            self.output_projection = None
    
    def encode(self, input):
        if self.use_projection:
            input = self.input_projection(input)
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        if self.output_projection is not None:
            result = self.output_projection(result)
        return result
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Improved loss function with numerical stability
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean vector from encoder
            logvar: Log variance vector from encoder
            
        Returns:
            loss: Total loss (reconstruction + KL divergence)
        """
        # Reconstruction loss (using mean squared error)
        # Use a smaller weight for the reconstruction term to balance with KL divergence
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        # Add small epsilon to avoid numerical instability
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply scaling factor to balance terms and handle extreme values
        # Add epsilon to avoid division by zero
        reconstruction_weight = 0.01  # Reduce weight of reconstruction term
        kl_weight = 1.0  # Weight for KL divergence term
        epsilon = 1e-8  # Small constant to avoid division by zero or extreme values
        
        # Prevent numerical instability for large KLD values
        if not torch.isfinite(KLD).all() or KLD > 1e6:
            print("Warning: KLD contains non-finite values or is extremely large. Clamping KLD.")
            KLD = torch.clamp(KLD, max=1e6)  # Clamp to reasonable maximum
        
        if torch.isnan(MSE).any() or torch.isnan(KLD).any():
            print("Warning: NaN values detected in loss. Using fallback values.")
            return torch.tensor(10.0, device=mu.device)  # Return a reasonable fallback value
        
        total_loss = (reconstruction_weight * MSE) + (kl_weight * KLD)
        
        # Handle edge cases
        if not torch.isfinite(total_loss):
            print("Warning: Total loss is not finite. Using fallback loss value.")
            return torch.tensor(10.0, device=mu.device)
            
        return total_loss

    def calculate_reconstruction_error(self, x):
        """Calculate reconstruction error for input x."""
        with torch.no_grad():
            recon_x, _, _ = self.forward(x)
        error = F.mse_loss(recon_x, x, reduction='mean')
        return error.item()

class GradientVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128, dropout_rate=0.2, projection_dim=None, use_batch_norm=True):
        """
        Enhanced Gradient VAE implementation with improved architecture
        """
        super(GradientVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_projection = projection_dim is not None
        
        if self.use_projection:
            self.actual_input_dim = projection_dim
            # Use more aggressive progressive dimension reduction
            reduction_stages = 6
            dims = [input_dim]
            for i in range(reduction_stages - 1):
                dims.append(dims[-1] // 8)  # More aggressive reduction
            dims.append(projection_dim)
            
            # Create memory-efficient projection layers with batch norm
            projection_layers = []
            for i in range(len(dims) - 1):
                projection_layers.extend([
                    nn.Linear(dims[i], dims[i+1]),
                    nn.LayerNorm(dims[i+1]) if use_batch_norm else nn.Identity(),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout_rate)
                ])
            self.projection = nn.Sequential(*projection_layers)
            
            # Create memory-efficient inverse projection layers with batch norm
            inverse_projection_layers = []
            dims.reverse()
            for i in range(len(dims) - 1):
                inverse_projection_layers.extend([
                    nn.Linear(dims[i], dims[i+1]),
                    nn.LayerNorm(dims[i+1]) if use_batch_norm else nn.Identity(),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout_rate)
                ])
            self.inverse_projection = nn.Sequential(*inverse_projection_layers)
            
            print("\nVAE Memory-Efficient Configuration:")
            total_params = sum(dims[i] * dims[i+1] for i in range(len(dims) - 1))
            print(f"Total parameters: {total_params:,}")
            print(f"Estimated memory: {(total_params * 4) / (1024*1024):.2f} MB")
            print("Progressive dimension reduction:")
            for i, dim in enumerate(dims):
                print(f"  Layer {i}: {dim:,} dimensions")
            
        else:
            self.actual_input_dim = input_dim
            
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.actual_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, self.actual_input_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.numel() > 0 and module.weight.dim() > 0:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    if fan_in != 0 and fan_out != 0:
                        torch.nn.init.xavier_uniform_(module.weight)
                    else:
                        # Fallback to a simpler initialization for degenerate cases
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
                # Check if module has bias attribute and it's not None
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def encode(self, x):
        # Process in chunks if input is too large
        if self.use_projection:
            x = self.projection(x)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def decode(self, z):
        h = self.decoder(z)
        if self.use_projection:
            h = self.inverse_projection(h)
        return h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        Improved loss function with numerical stability
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean vector from encoder
            logvar: Log variance vector from encoder
            
        Returns:
            loss: Total loss (reconstruction + KL divergence)
        """
        # Reconstruction loss (using mean squared error)
        # Use a smaller weight for the reconstruction term to balance with KL divergence
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        # Add small epsilon to avoid numerical instability
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply scaling factor to balance terms and handle extreme values
        # Add epsilon to avoid division by zero
        reconstruction_weight = 0.01  # Reduce weight of reconstruction term
        kl_weight = 1.0  # Weight for KL divergence term
        epsilon = 1e-8  # Small constant to avoid division by zero or extreme values
        
        # Prevent numerical instability for large KLD values
        if not torch.isfinite(KLD).all() or KLD > 1e6:
            print("Warning: KLD contains non-finite values or is extremely large. Clamping KLD.")
            KLD = torch.clamp(KLD, max=1e6)  # Clamp to reasonable maximum
        
        if torch.isnan(MSE).any() or torch.isnan(KLD).any():
            print("Warning: NaN values detected in loss. Using fallback values.")
            return torch.tensor(10.0, device=mu.device)  # Return a reasonable fallback value
        
        total_loss = (reconstruction_weight * MSE) + (kl_weight * KLD)
            
        # Handle edge cases
        if not torch.isfinite(total_loss):
            print("Warning: Total loss is not finite. Using fallback loss value.")
            return torch.tensor(10.0, device=mu.device)
            
        return total_loss
    
    def calculate_reconstruction_error(self, x):
        """Calculate reconstruction error for input x."""
        with torch.no_grad():
            recon_x, _, _ = self.forward(x)
        error = F.mse_loss(recon_x, x, reduction='mean')
        return error.item()
        
    def get_reconstruction_error(self, x):
        """Alias for calculate_reconstruction_error for backward compatibility"""
        with torch.no_grad():
            self.eval()  # Set model to evaluation mode
            if self.use_projection:
                x_projected = self.projection(x)
            else:
                x_projected = x
            recon_x, _, _ = self.forward(x)
            recon_error = F.mse_loss(recon_x, x_projected, reduction='sum').item()
            normalized_error = recon_error / x_projected.size(1)
            return normalized_error 