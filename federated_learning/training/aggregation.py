import torch
import torch.nn as nn
import numpy as np
from federated_learning.config.config import *

def extract_gradient_features(gradients, root_gradient=None):
    """
    Extract meaningful features from gradients for analysis.
    Enhanced with better normalization and pattern detection.
    
    Args:
        gradients: List of client gradients
        root_gradient: Optional reference gradient
        
    Returns:
        features: Tensor of gradient features
    """
    features = []
    
    # If no root gradient provided, use mean of gradients
    if root_gradient is None:
        root_gradient = torch.mean(torch.stack(gradients), dim=0)
    
    # Get root gradient statistics
    root_norm = torch.norm(root_gradient)
    root_mean = torch.mean(root_gradient).item()
    root_std = torch.std(root_gradient).item()
    
    # Get gradient norms before normalization
    grad_norms = [torch.norm(grad) for grad in gradients]
    max_norm = max(grad_norms)
    
    # Normalize gradients for directional comparison
    normalized_root = root_gradient / root_norm if root_norm > 0 else root_gradient
    normalized_grads = []
    for grad in gradients:
        grad_norm = torch.norm(grad)
        if grad_norm > 0:
            normalized_grads.append(grad / grad_norm)
        else:
            normalized_grads.append(grad)
    
    for grad, norm, norm_grad in zip(gradients, grad_norms, normalized_grads):
        # Scale-based features
        norm_ratio = norm / max_norm if max_norm > 0 else 0.0
        relative_norm = norm / root_norm if root_norm > 0 else norm
        
        # Direction-based features
        cosine_sim = torch.nn.functional.cosine_similarity(
            norm_grad.view(1, -1), 
            normalized_root.view(1, -1)
        ).item()
        
        # L2 distance in normalized space
        l2_dist = torch.norm(norm_grad - normalized_root).item()
        
        # Statistical features with improved normalization
        mean = torch.mean(grad).item()
        std = torch.std(grad).item()
        
        # Distribution features with better handling of edge cases
        if std > 0:
            kurtosis = torch.mean((grad - mean)**4) / (std**4)
            skewness = torch.mean((grad - mean)**3) / (std**3)
        else:
            kurtosis = torch.tensor(0.0)
            skewness = torch.tensor(0.0)
        
        # Enhanced pattern analysis
        abs_grad = torch.abs(grad)
        max_abs = torch.max(abs_grad).item()
        min_abs = torch.min(abs_grad[abs_grad > 0]).item() if torch.any(abs_grad > 0) else 0.0
        max_min_ratio = max_abs/min_abs if min_abs > 0 else max_abs
        
        # Deviation from root with improved normalization
        mean_dev = abs(mean - root_mean) / (abs(root_mean) if root_mean != 0 else 1.0)
        std_dev = abs(std - root_std) / (root_std if root_std != 0 else 1.0)
        
        # Additional pattern detection features
        # 1. Gradient sparsity
        sparsity = (abs_grad < 1e-6).float().mean().item()
        
        # 2. Gradient distribution skew
        sorted_grad = torch.sort(abs_grad)[0]
        q1_idx = int(0.25 * len(sorted_grad))
        q3_idx = int(0.75 * len(sorted_grad))
        iqr = sorted_grad[q3_idx] - sorted_grad[q1_idx]
        skew_ratio = iqr / (max_abs - min_abs) if (max_abs - min_abs) > 0 else 0.0
        
        # 3. Pattern consistency
        pattern_consistency = torch.corrcoef(
            torch.stack([abs_grad, torch.abs(root_gradient)])
        )[0, 1].item()
        
        # 4. Gradient stability
        stability = 1.0 - (std / (abs(mean) + 1e-6))
        
        features.append([
            cosine_sim,          # Direction similarity (-1 to 1)
            norm_ratio,          # Scale relative to max (0 to 1)
            relative_norm,       # Scale relative to root (0 to inf)
            l2_dist,            # Normalized space distance (0 to 2)
            mean_dev,           # Mean deviation (0 to inf)
            std_dev,            # Std deviation (0 to inf)
            kurtosis.item(),    # Distribution shape
            skewness.item(),    # Distribution asymmetry
            max_min_ratio,      # Value range indicator
            max_abs,            # Absolute magnitude
            sparsity,           # Gradient sparsity (0 to 1)
            skew_ratio,         # Distribution skew ratio (0 to 1)
            pattern_consistency, # Pattern consistency (-1 to 1)
            stability           # Gradient stability (0 to 1)
        ])
    
    return torch.tensor(features, dtype=torch.float32)

def aggregate_gradients(client_gradients, aggregation_method='fedavg', **kwargs):
    """
    Aggregate gradients using specified method.
    
    Args:
        client_gradients: List of client gradients
        aggregation_method: Method to use for aggregation
        **kwargs: Additional arguments for specific methods
        
    Returns:
        aggregated_gradient: Aggregated gradient
    """
    # Convert to tensor if needed
    if not isinstance(client_gradients[0], torch.Tensor):
        client_gradients = [torch.tensor(g) for g in client_gradients]
    
    if aggregation_method == 'fedavg':
        # Simple averaging
        return torch.mean(torch.stack(client_gradients), dim=0)
    
    elif aggregation_method == 'fedprox':
        # FedProx - Same aggregation as FedAvg but with proximal term added during client training
        return torch.mean(torch.stack(client_gradients), dim=0)
    
    elif aggregation_method == 'fedadmm':
        # FedADMM - ADMM-based aggregation with additional constraints
        # Parameters
        rho = kwargs.get('rho', FEDADMM_RHO)
        sigma = kwargs.get('sigma', FEDADMM_SIGMA)
        iterations = kwargs.get('iterations', FEDADMM_ITERATIONS)
        
        # Initial aggregation is average gradient
        z = torch.mean(torch.stack(client_gradients), dim=0)
        
        # Get dual variables if available, or initialize them
        dual_variables = kwargs.get('dual_variables')
        if dual_variables is None:
            dual_variables = [torch.zeros_like(z) for _ in range(len(client_gradients))]
        
        # Iterative optimization
        for _ in range(iterations):
            # Update z (consensus variable)
            sum_term = torch.zeros_like(z)
            for i, grad in enumerate(client_gradients):
                sum_term += grad + dual_variables[i]
            z = sum_term / (len(client_gradients) + rho)
            
            # Update dual variables (one per client)
            for i, grad in enumerate(client_gradients):
                dual_variables[i] = dual_variables[i] + sigma * (grad - z)
        
        return z
    
    elif aggregation_method == 'fedbn':
        # FedBN - Skip updating batch normalization layers
        # Model architecture is needed for proper BN filtering, so we use model parameter names
        model = kwargs.get('model')
        
        # If model is provided, we can do accurate BN filtering
        if model is not None:
            # Collect all batch normalization parameter names
            bn_params = set()
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm3d):
                    # Add all parameters for this BatchNorm layer
                    for child_name, _ in module.named_parameters(recurse=False):
                        full_name = f"{name}.{child_name}"
                        bn_params.add(full_name)
                    # Also add buffers (running_mean, running_var, etc.)
                    for child_name, _ in module.named_buffers(recurse=False):
                        full_name = f"{name}.{child_name}"
                        bn_params.add(full_name)
            
            # Also catch BatchNorm from other naming patterns
            for name, param in model.named_parameters():
                if '.bn' in name or 'downsample.1' in name or name.endswith('bn.weight') or name.endswith('bn.bias'):
                    bn_params.add(name)
            
            print(f"FedBN: Identified {len(bn_params)} BatchNorm parameters to preserve")
                    
            # Get parameter shapes and create a mapping of parameters to indices in the flattened gradient
            param_shapes = {}
            param_is_bn = {}
            param_indices = {}
            
            offset = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_shapes[name] = param.shape
                    is_bn = name in bn_params
                    param_is_bn[name] = is_bn
                    
                    # Store the start and end indices for this parameter in the flattened gradient
                    size = param.numel()
                    param_indices[name] = (offset, offset + size)
                    offset += size
            
            # Validate gradient size
            expected_gradient_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            if client_gradients[0].numel() != expected_gradient_size:
                print(f"Warning: Gradient size mismatch. Expected {expected_gradient_size} but got {client_gradients[0].numel()}")
                # Fall back to basic averaging
                return torch.mean(torch.stack(client_gradients), dim=0)
            
            # Create a mask for non-BN parameters
            gradient_mask = torch.ones_like(client_gradients[0])
            for name, (start_idx, end_idx) in param_indices.items():
                if param_is_bn[name]:
                    gradient_mask[start_idx:end_idx] = 0.0
            
            # Apply mask to each client gradient
            masked_gradients = [grad * gradient_mask for grad in client_gradients]
            
            # Compute mean of masked gradients
            mean_gradient = torch.mean(torch.stack(masked_gradients), dim=0)
            
            # Create weights if provided for weighted averaging
            weights = kwargs.get('weights')
            if weights is not None:
                # Get weighted sum of gradients except for BN parameters
                weighted_gradient = torch.zeros_like(mean_gradient)
                for i, grad in enumerate(masked_gradients):
                    weighted_gradient += grad * weights[i]
                
                return weighted_gradient
            
            return mean_gradient
        
        # If model is not provided, use basic stacking method (less accurate for FedBN)
        else:
            print("Warning: Model not provided for FedBN. Using basic gradient averaging.")
            return torch.mean(torch.stack(client_gradients), dim=0)
    
    elif aggregation_method == 'fedbn_fedprox':
        # Combination of FedBN and FedProx
        # Same aggregation as FedBN but with proximal term during client training
        model = kwargs.get('model')
        
        # Reuse the FedBN implementation for consistency
        return aggregate_gradients(
            client_gradients=client_gradients,
            aggregation_method='fedbn',
            model=model,
            weights=kwargs.get('weights')
        )
    
    elif aggregation_method == 'fednova':
        # FedNova - Normalized averaging based on local steps
        client_steps = kwargs.get('client_steps')
        if client_steps is None:
            # If steps not provided, assume equal number of steps
            client_steps = [1.0] * len(client_gradients)
        
        # Normalize by steps
        normalized_gradients = []
        for grad, steps in zip(client_gradients, client_steps):
            if steps > 0:
                normalized_gradients.append(grad / steps)
            else:
                normalized_gradients.append(grad)
        
        return torch.mean(torch.stack(normalized_gradients), dim=0)
    
    elif aggregation_method == 'feddwa':
        # FedDWA - Dynamic weighted aggregation based on client performance
        client_metrics = kwargs.get('client_metrics')
        weighting_method = kwargs.get('weighting_method', FEDDWA_WEIGHTING)
        history_factor = kwargs.get('history_factor', FEDDWA_HISTORY_FACTOR)
        
        if client_metrics is None:
            # If metrics not provided, use equal weighting
            return torch.mean(torch.stack(client_gradients), dim=0)
        
        # Convert to tensor if needed
        metrics_tensor = torch.tensor(client_metrics, dtype=torch.float32)
        
        # Handle different types of metrics according to weighting method
        if weighting_method == 'accuracy':
            # Higher is better, use directly
            weights_raw = metrics_tensor
        elif weighting_method == 'loss':
            # Lower is better, invert the scale
            weights_raw = 1.0 / (metrics_tensor + 1e-5)
        elif weighting_method == 'gradient_norm':
            # Use gradient norm to indicate client importance
            norms = torch.tensor([torch.norm(g).item() for g in client_gradients])
            # Clip extremely large norms
            max_norm = norms.mean() * 2.0
            norms = torch.clamp(norms, max=max_norm)
            weights_raw = norms
        else:
            # Default to equal weighting
            weights_raw = torch.ones_like(metrics_tensor)
            
        # Apply softmax with temperature for smoother weights
        temp = 2.0  # Temperature parameter
        exp_weights = torch.exp(weights_raw / temp)
        weights = exp_weights / torch.sum(exp_weights)
        
        # Apply history factor if previous weights are provided
        prev_weights = kwargs.get('prev_weights')
        if prev_weights is not None and history_factor > 0:
            # Convert to tensor if needed
            if not isinstance(prev_weights, torch.Tensor):
                prev_weights = torch.tensor(prev_weights, dtype=torch.float32)
            # Blend current and previous weights
            weights = (1 - history_factor) * weights + history_factor * prev_weights
            # Renormalize
            weights = weights / torch.sum(weights)
        
        # Ensure weights sum to 1
        weights = weights / torch.sum(weights)
        
        # Apply dynamic weights
        weighted_sum = torch.zeros_like(client_gradients[0])
        for i, grad in enumerate(client_gradients):
            weighted_sum += grad * weights[i]
        
        return weighted_sum
    
    elif aggregation_method == 'direct':
        # Get necessary arguments
        root_gradient = kwargs.get('root_gradient')
        dual_attention = kwargs.get('dual_attention')
        
        if dual_attention is None:
            raise ValueError("DualAttention model must be provided for direct aggregation")
        
        # Extract features
        features = extract_gradient_features(client_gradients, root_gradient)
        
        # Get weights from DualAttention
        weights = dual_attention.get_gradient_weights(features)
        
        # Apply weights
        weighted_sum = torch.zeros_like(client_gradients[0])
        for grad, weight in zip(client_gradients, weights):
            weighted_sum += grad * weight
        
        return weighted_sum
    
    elif aggregation_method == 'weighted':
        # Use provided weights
        weights = kwargs.get('weights')
        if weights is None:
            raise ValueError("Weights must be provided for weighted aggregation")
        
        weighted_sum = torch.zeros_like(client_gradients[0])
        for grad, weight in zip(client_gradients, weights):
            weighted_sum += grad * weight
        
        return weighted_sum
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

def analyze_gradient_weights(weights, features):
    """
    Analyze the relationship between gradient features and assigned weights.
    
    Args:
        weights: Tensor of weights assigned to gradients
        features: Tensor of gradient features
        
    Returns:
        analysis: Dictionary containing analysis results
    """
    analysis = {}
    
    # Correlation between weights and features
    for i in range(features.shape[1]):
        correlation = torch.corrcoef(
            torch.stack([weights, features[:, i]])
        )[0, 1].item()
        analysis[f'feature_{i}_correlation'] = correlation
    
    # Weight statistics
    analysis['weight_mean'] = weights.mean().item()
    analysis['weight_std'] = weights.std().item()
    analysis['weight_min'] = weights.min().item()
    analysis['weight_max'] = weights.max().item()
    
    # Feature importance based on correlation
    feature_importance = torch.abs(torch.tensor([
        analysis[f'feature_{i}_correlation'] for i in range(features.shape[1])
    ]))
    analysis['most_important_features'] = torch.argsort(feature_importance, descending=True).tolist()
    
    return analysis 