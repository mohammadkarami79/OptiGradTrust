"""
Utility functions for gradient feature extraction.
"""
import torch
import torch.nn.functional as F
import numpy as np

def compute_gradient_features(grad, raw_grad, vae, root_gradients=None, client_gradients=None, all_raw_gradients=None, typical_RE=None):
    """
    Compute feature vector for a gradient with standardized 5 features.
    
    Args:
        grad: Normalized gradient
        raw_grad: Raw gradient before normalization
        vae: Trained VAE model
        root_gradients: List of root gradients for comparison
        client_gradients: List of other client gradients for computing neighbor similarity
        all_raw_gradients: List of all raw gradients for adaptive norm scaling
        typical_RE: Typical reconstruction error value for normalization
        
    Returns:
        Tensor with 5 features:
        1. Reconstruction error from VAE (normalized)
        2. Cosine similarity to root gradients (normalized)
        3. Cosine similarity to other clients (normalized)
        4. Gradient norm (normalized adaptively)
        5. Pattern consistency with root gradients (normalized)
    """
    # Determine the device used by the VAE
    vae_device = next(vae.parameters()).device

    # Transfer gradient to the VAE's device for computation
    vae_grad = grad.to(vae_device)
    
    # Ensure gradient has batch dimension
    if vae_grad.dim() == 1:
        vae_grad = vae_grad.unsqueeze(0)

    # 1. Compute reconstruction error using VAE
    try:
        with torch.no_grad():
            # Set VAE to eval mode to avoid batch norm/dropout issues
            vae.eval()
            
            # Run VAE forward pass
            recon_batch, mu, logvar = vae(vae_grad)
            
            # Calculate MSE directly with input and output
            RE_val = F.mse_loss(recon_batch, vae_grad, reduction='sum').item()

            # Normalize RE_val to a 0-1 range (higher is more anomalous)
            # Use adaptive normalization based on typical reconstruction errors
            if typical_RE is not None and typical_RE > 0:
                RE_val_normalized = RE_val / (typical_RE * 2)  # Normalize relative to typical values
            else:
                RE_val_normalized = min(RE_val / 10.0, 1.0)  # Fallback normalization
            
            # Lower is better for reconstruction error, so invert for consistency
            # Now 0 means high error (bad) and 1 means low error (good)
            RE_val_normalized = 1.0 - min(RE_val_normalized, 1.0)
    except Exception as e:
        print(f"Error in computing VAE features: {str(e)}")
        # Provide a fallback value - 0 means high error (suspicious gradient)
        RE_val_normalized = 0.0
    
    # 2. Compute mean cosine similarity with root gradients
    if root_gradients is not None and len(root_gradients) > 0:
        cos_root_vals = []
        for r in root_gradients:
            # Ensure both tensors are on the same device
            r_device = r.to(grad.device)
            cos_sim = F.cosine_similarity(grad.flatten(), r_device.flatten(), dim=0).item()
            cos_root_vals.append(cos_sim)
        
        mean_cosine_root = np.mean(cos_root_vals)
        # Normalize cosine similarity from [-1, 1] to [0, 1] range
        # Higher is better (more similar to trusted root gradients)
        mean_cosine_root_normalized = (mean_cosine_root + 1) / 2.0
    else:
        mean_cosine_root_normalized = 0.5  # Neutral value if no root gradients
    
    # 3. Compute mean similarity to other clients
    if client_gradients is not None and len(client_gradients) > 1:  # Need at least 2 clients
        cos_client_vals = []
        for client_grad in client_gradients:
            if not torch.equal(client_grad, grad):  # Skip self-comparison
                client_grad_device = client_grad.to(grad.device)
                cos_sim = F.cosine_similarity(grad.flatten(), client_grad_device.flatten(), dim=0).item()
                cos_client_vals.append(cos_sim)

        mean_neighbor_sim = np.mean(cos_client_vals) if cos_client_vals else 0.0
        # Normalize from [-1, 1] to [0, 1]
        mean_neighbor_sim_normalized = (mean_neighbor_sim + 1) / 2.0
    else:
        mean_neighbor_sim_normalized = 0.5  # Neutral value if no other clients
    
    # 4. Compute norm of raw gradient with adaptive normalization
    grad_norm = torch.norm(raw_grad).item()
    
    if all_raw_gradients is not None and len(all_raw_gradients) > 0:
        # Compute statistics of all gradient norms for better normalization
        all_norms = [torch.norm(g).item() for g in all_raw_gradients]
        median_norm = np.median(all_norms)
        max_norm = max(all_norms)
        
        # Normalize relative to median and max for better discrimination
        if median_norm > 0:
            # This preserves the relative differences between gradient norms
            # while mapping most values to a reasonable range
            grad_norm_normalized = min(grad_norm / (median_norm * 2), 1.0)
        else:
            grad_norm_normalized = min(grad_norm / max(max_norm, 1e-8), 1.0)
    else:
        # Fallback normalization if we don't have other gradients to compare
        grad_norm_normalized = min(grad_norm / 10.0, 1.0)
    
    # 5. Pattern consistency feature
    # Measures how consistent the gradient pattern is with root gradients
    if root_gradients is not None and len(root_gradients) > 0:
        # Use the mean of root gradients as reference
        root_mean = torch.stack([r.to(grad.device) for r in root_gradients]).mean(dim=0)
        
        # Calculate pattern consistency using correlation of signs
        grad_signs = torch.sign(grad.flatten())
        root_signs = torch.sign(root_mean.flatten())
        
        # Calculate percentage of matching signs (ranges from 0 to 1)
        matching_signs = (grad_signs == root_signs).float().mean().item()
        
        # Additional pattern consistency using distribution similarity
        # Compare sorted absolute values to see if distribution shapes match
        grad_abs = torch.abs(grad.flatten())
        root_abs = torch.abs(root_mean.flatten())
        
        # Sort both and sample points for comparison
        n_samples = min(1000, grad_abs.numel())
        indices = torch.linspace(0, grad_abs.numel()-1, n_samples).long()
        
        grad_abs_sorted = torch.sort(grad_abs)[0][indices]
        root_abs_sorted = torch.sort(root_abs)[0][indices]
        
        # Normalize both to [0,1] for shape comparison only
        grad_abs_norm = grad_abs_sorted / (torch.max(grad_abs_sorted) + 1e-8)
        root_abs_norm = root_abs_sorted / (torch.max(root_abs_sorted) + 1e-8)
        
        # Calculate distribution similarity (higher is better)
        dist_similarity = 1.0 - torch.mean(torch.abs(grad_abs_norm - root_abs_norm)).item()
        
        # Combine both metrics for overall pattern consistency
        pattern_consistency = 0.5 * matching_signs + 0.5 * dist_similarity
    else:
        pattern_consistency = 0.5  # Neutral value if no root gradients
    
    # Combine all features into a single tensor
    features = torch.tensor([
        RE_val_normalized,           # Reconstruction error (0=bad, 1=good)
        mean_cosine_root_normalized, # Root similarity (0=dissimilar, 1=similar)
        mean_neighbor_sim_normalized, # Client similarity (0=dissimilar, 1=similar)
        grad_norm_normalized,        # Gradient norm (0=small, 1=large)
        pattern_consistency          # Pattern consistency (0=inconsistent, 1=consistent)
    ], device=grad.device)
    
    return features

def normalize_features(features):
    """
    Normalize feature vectors to ensure they are in [0, 1] range and properly scaled.
    This enhanced version preserves relative differences while ensuring proper scaling.
    
    Args:
        features: Tensor of feature vectors [batch_size, feature_dim]
        
    Returns:
        normalized_features: Normalized feature vectors [batch_size, feature_dim]
    """
    # Ensure features are in [0, 1] range
    features = torch.clamp(features, 0.0, 1.0)
    
    # If features is a batch, normalize each feature dimension independently
    if features.dim() > 1:
        # Get min and max for each feature dimension
        min_vals, _ = torch.min(features, dim=0, keepdim=True)
        max_vals, _ = torch.max(features, dim=0, keepdim=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-8] = 1.0
        
        # Normalize to [0, 1] range while preserving relative differences
        normalized_features = (features - min_vals) / range_vals
        
        # For features where higher is better (1, 2, 4), keep as is
        # For features where lower is better (0, 3), we've already inverted them in compute_gradient_features
        
        # Apply non-linear scaling to enhance small differences if needed
        # This makes the distribution more discriminative
        # Applying mild sigmoid scaling centered at 0.5
        normalized_features = torch.sigmoid((normalized_features - 0.5) * 4) 
    else:
        # Single feature vector
        normalized_features = features
    
    return normalized_features 