import torch
import torch.nn.functional as F
import numpy as np
from federated_learning.config.config import *

def simulate_attack(raw_grad, attack_type, attack_params=None):
    """
    Simulates different types of attacks on the gradient.
    
    Args:
        raw_grad: The raw gradient to be attacked
        attack_type: Type of attack to simulate
        attack_params: Optional dictionary of attack parameters
        
    Returns:
        Modified gradient after the attack
    """
    device = raw_grad.device
    
    # Use attack parameters from config or provided params
    if attack_params is None:
        attack_params = {}
        
    scaling_factor = attack_params.get('scaling_factor', SCALING_FACTOR if 'SCALING_FACTOR' in globals() else 15.0)
    partial_percent = attack_params.get('partial_percent', PARTIAL_SCALING_PERCENT if 'PARTIAL_SCALING_PERCENT' in globals() else 0.3)

    # Ensure raw_grad is a PyTorch tensor
    if not isinstance(raw_grad, torch.Tensor):
        print("Warning: raw_grad is not a torch.Tensor. Converting...")
        raw_grad = torch.tensor(raw_grad, device=device)

    # Store original gradient norm for reporting
    original_norm = torch.norm(raw_grad).item()
    
    # If attack type is 'none', return the original gradient without modification
    if attack_type == 'none':
        print("No attack applied (attack_type is 'none')")
        return raw_grad

    print(f"Simulating {attack_type} attack on device: {device}")
    print(f"Original gradient norm: {original_norm:.4f}")

    modified_grad = None
    
    if attack_type == 'label_flipping' or attack_type == 'sign_flipping_attack':
        # Inverts the gradient direction to simulate the effect of flipping labels
        modified_grad = -raw_grad
        attack_desc = "Sign flipping attack"
        
    elif attack_type == 'scaling_attack':
        # Scales the gradient by a factor to amplify its effect
        modified_grad = raw_grad * scaling_factor
        attack_desc = f"Scaling attack (factor: {scaling_factor:.1f})"
        
    elif attack_type == 'partial_scaling_attack':
        # More sophisticated than scaling attack - only scales a subset of gradient values
        # Creates a random mask where a percentage of elements are affected
        mask = torch.zeros_like(raw_grad, device=device)
        indices = torch.randperm(raw_grad.numel(), device=device)[:int(partial_percent * raw_grad.numel())]
        mask.view(-1)[indices] = 1.0
        
        # Apply partial scaling: Elements where mask is 1 are scaled, others remain unchanged
        modified_grad = raw_grad * (1 - mask) + raw_grad * mask * scaling_factor
        attack_desc = f"Partial scaling attack (factor: {scaling_factor:.1f}, {partial_percent*100:.0f}% of elements)"
        
    elif attack_type == 'backdoor_attack':
        # Adds a constant value to the gradient
        modified_grad = raw_grad + torch.ones_like(raw_grad) * scaling_factor * 0.1
        attack_desc = "Backdoor attack (constant bias addition)"
        
    elif attack_type == 'noise_attack':
        # Adds random noise to the gradient 
        noise_level = 0.1 * torch.norm(raw_grad)
        modified_grad = raw_grad + noise_level * torch.randn_like(raw_grad)
        attack_desc = "Noise attack (random noise addition)"
        
    elif attack_type == 'min_max_attack':
        # Min-max attack as described in FLTrust paper
        # First normalize the gradient to unit norm to avoid magnitude-based detection
        normalized_grad = raw_grad / (torch.norm(raw_grad) + 1e-8)
        # Scale the normalized gradient to maximize impact
        modified_grad = -normalized_grad * scaling_factor
        attack_desc = "Min-max attack (normalized negation)"
        
    elif attack_type == 'min_sum_attack':
        # Min-sum attack as described in FLTrust paper
        normalized_grad = raw_grad / (torch.norm(raw_grad) + 1e-8)
        modified_grad = -normalized_grad * scaling_factor
        attack_desc = "Min-sum attack (normalized negation)"
        
    elif attack_type == 'alternating_attack':
        # Alternating attack - oscillating between positive and negative values
        sign_mask = torch.ones_like(raw_grad)
        # Create an alternating pattern of +1 and -1
        for i in range(sign_mask.numel()):
            if i % 2 == 0:
                sign_mask.view(-1)[i] = 1
            else:
                sign_mask.view(-1)[i] = -1
        modified_grad = raw_grad * sign_mask * scaling_factor
        attack_desc = "Alternating attack (oscillating pattern)"
        
    elif attack_type == 'targeted_attack':
        # Targeted attack aims to introduce a specific pattern into the model
        mask = torch.zeros_like(raw_grad)
        # Target first 10% of parameters
        num_elements = int(0.1 * raw_grad.numel())
        mask.view(-1)[:num_elements] = 1
        modified_grad = raw_grad * (1 - mask) + mask * scaling_factor * torch.randn_like(raw_grad)
        attack_desc = "Targeted attack (10% of parameters)"
        
    elif attack_type == 'gradient_inversion_attack':
        # Gradient inversion attack with different magnitudes for different parts
        quarter_size = raw_grad.numel() // 4
        modified_grad = raw_grad.clone()

        # First quarter: regular inversion
        modified_grad.view(-1)[:quarter_size] = -raw_grad.view(-1)[:quarter_size]

        # Second quarter: amplified inversion
        modified_grad.view(-1)[quarter_size:2*quarter_size] = -2 * raw_grad.view(-1)[quarter_size:2*quarter_size]

        # Third quarter: reduced inversion
        modified_grad.view(-1)[2*quarter_size:3*quarter_size] = -0.5 * raw_grad.view(-1)[2*quarter_size:3*quarter_size]

        # Fourth quarter: unchanged but amplified
        modified_grad.view(-1)[3*quarter_size:] = 3 * raw_grad.view(-1)[3*quarter_size:]
        
        attack_desc = "Gradient inversion attack (varied patterns)"

    else:
        # No attack
        print("No attack applied (attack_type not recognized)")
        return raw_grad

    # Calculate and print attack metrics
    modified_norm = torch.norm(modified_grad).item()
    norm_change_abs = modified_norm - original_norm
    norm_change_pct = (modified_norm / original_norm - 1) * 100
    
    cosine_sim = F.cosine_similarity(
        modified_grad.view(1, -1), 
        raw_grad.view(1, -1)
    ).item()
    
    print(f"\n=== {attack_desc} ===")
    print(f"Original gradient norm: {original_norm:.4f}")
    print(f"Modified gradient norm: {modified_norm:.4f}")
    print(f"Absolute change: {norm_change_abs:.4f}")
    print(f"Relative change: {norm_change_pct:.2f}%")
    print(f"Cosine similarity: {cosine_sim:.4f}")
    
    return modified_grad

def detect_gradient_anomalies(gradients, client_indices=None):
    """
    Detect potential anomalies in gradients that could indicate attacks.
    
    Args:
        gradients: List of gradient tensors
        client_indices: Optional list of client indices corresponding to gradients
        
    Returns:
        Dictionary mapping client indices to their anomaly types
    """
    if client_indices is None:
        client_indices = list(range(len(gradients)))
        
    if len(gradients) == 0:
        return {}
    
    # Calculate gradient norms
    norms = [torch.norm(grad).item() for grad in gradients]
    
    # Basic statistics
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    # Detect anomalies
    anomalies = {}
    
    # Check for abnormally high norms (potential scaling attacks)
    high_norm_threshold = mean_norm + 1.5 * std_norm
    for i, (client_idx, norm) in enumerate(zip(client_indices, norms)):
        anomaly_types = []
        
        # Check for high norm (potential scaling attack)
        if norm > high_norm_threshold:
            anomaly_types.append({
                'type': 'high_norm',
                'value': norm,
                'threshold': high_norm_threshold,
                'deviation': (norm - mean_norm) / std_norm
            })
            
        # Check cosine similarity with other gradients
        similarity_scores = []
        for j, other_grad in enumerate(gradients):
            if i != j:
                similarity = F.cosine_similarity(
                    gradients[i].view(1, -1), other_grad.view(1, -1)
                ).item()
                similarity_scores.append(similarity)
                
        # If this client's gradient is very different from others
        if similarity_scores:
            mean_similarity = np.mean(similarity_scores)
            if mean_similarity < 0.2:  # Very low similarity threshold
                anomaly_types.append({
                    'type': 'low_similarity',
                    'value': mean_similarity,
                    'threshold': 0.2
                })
        
        # Only add to anomalies if we found something suspicious
        if anomaly_types:
            anomalies[client_idx] = anomaly_types
    
    return anomalies

def analyze_attack_characteristics(original_grad, attacked_grad):
    """
    Analyze the characteristics of an attack by comparing original and attacked gradients.
    
    Args:
        original_grad: Original gradient tensor before attack
        attacked_grad: Modified gradient tensor after attack
        
    Returns:
        Dictionary of attack characteristics
    """
    if not isinstance(original_grad, torch.Tensor) or not isinstance(attacked_grad, torch.Tensor):
        raise TypeError("Both gradients must be torch tensors")
        
    if original_grad.shape != attacked_grad.shape:
        raise ValueError("Gradient shapes must match")
        
    # Basic metrics
    orig_norm = torch.norm(original_grad).item()
    attacked_norm = torch.norm(attacked_grad).item()
    norm_ratio = attacked_norm / orig_norm
    
    # Cosine similarity 
    cosine_sim = F.cosine_similarity(
        original_grad.view(1, -1), 
        attacked_grad.view(1, -1)
    ).item()
    
    # Element-wise differences
    diff = attacked_grad - original_grad
    diff_norm = torch.norm(diff).item()
    
    # Distribution of changes
    abs_diff = torch.abs(diff)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    # Estimate of affected elements 
    # (elements with significant change, defined as > 3x the mean absolute difference)
    significant_diff_mask = abs_diff > (3 * mean_diff)
    affected_elements = torch.sum(significant_diff_mask).item()
    affected_percentage = affected_elements / original_grad.numel() * 100
    
    # For elements with significant change, calculate average scaling factor
    if torch.sum(significant_diff_mask) > 0:
        # Avoid division by zero
        original_significant = torch.where(
            torch.logical_and(significant_diff_mask, torch.abs(original_grad) > 1e-10),
            original_grad,
            torch.ones_like(original_grad) * 1e-10
        )
        
        attacked_significant = torch.where(
            significant_diff_mask, 
            attacked_grad,
            torch.zeros_like(attacked_grad)
        )
        
        # Calculate element-wise ratio
        element_ratio = attacked_significant / original_significant
        
        # Get mean ratio (excluding extremes)
        sorted_ratios = torch.sort(torch.abs(element_ratio.view(-1)))[0]
        middle_indices = sorted_ratios.shape[0] // 10
        if middle_indices > 0:
            estimated_scaling = sorted_ratios[middle_indices:-middle_indices].mean().item()
        else:
            estimated_scaling = sorted_ratios.mean().item()
    else:
        estimated_scaling = 1.0
    
    # Results
    results = {
        'original_norm': orig_norm,
        'attacked_norm': attacked_norm,
        'norm_ratio': norm_ratio,
        'cosine_similarity': cosine_sim,
        'diff_norm': diff_norm,
        'max_element_diff': max_diff,
        'mean_element_diff': mean_diff,
        'affected_elements': affected_elements,
        'affected_percentage': affected_percentage,
        'estimated_scaling_factor': estimated_scaling,
        'likely_attack_type': classify_attack(norm_ratio, cosine_sim, affected_percentage)
    }
    
    return results

def classify_attack(norm_ratio, cosine_sim, affected_percentage):
    """
    Classify likely attack type based on gradient analysis.
    
    Args:
        norm_ratio: Ratio of attacked norm to original norm
        cosine_sim: Cosine similarity between original and attacked gradient 
        affected_percentage: Percentage of significantly affected elements
        
    Returns:
        String describing likely attack type
    """
    # Sign flipping has negative cosine similarity
    if cosine_sim < -0.5:
        return "sign_flipping_attack"
        
    # Scaling attack has high norm ratio but preserves direction (high cosine similarity)
    if norm_ratio > 5 and cosine_sim > 0.9:
        return "scaling_attack"
        
    # Partial scaling affects a subset of elements but still shows increased norm
    if norm_ratio > 2 and affected_percentage < 50 and cosine_sim > 0.7:
        return "partial_scaling_attack"
        
    # Noise attack has lower cosine similarity but not negative
    if cosine_sim < 0.8 and norm_ratio > 0.8 and norm_ratio < 3:
        return "noise_attack"
        
    # No clear attack pattern
    return "unknown" 