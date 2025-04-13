import torch
import torch.nn.functional as F
from federated_learning.config.config import *

def simulate_attack(raw_grad, attack_type):
    """
    Simulates different types of attacks on the gradient.
    
    Args:
        raw_grad: The raw gradient to be attacked
        attack_type: Type of attack to simulate
        
    Returns:
        Modified gradient after the attack
    """
    if attack_type == 'label_flipping':
        # Inverts the gradient direction to simulate the effect of flipping labels
        # This is equivalent to minimizing the negative loss function
        return -raw_grad
        
    elif attack_type == 'scaling_attack':
        # Scales the gradient by the number of clients to amplify its effect
        # This attack aims to dominate the aggregated gradient
        return raw_grad * NUM_CLIENTS
        
    elif attack_type == 'partial_scaling_attack':
        # More sophisticated than scaling attack - only scales a subset of gradient values
        # Creates a random mask where ~66% of elements are 1 and the rest are 0
        scaling_factor = NUM_CLIENTS
        mask = (torch.rand(raw_grad.shape, device=raw_grad.device) < 0.66).float()
        # Elements where mask is 1 are scaled, others remain unchanged
        return raw_grad * (mask * scaling_factor + (1 - mask))
        
    elif attack_type == 'backdoor_attack':
        # Adds a constant value to the gradient
        # This creates a consistent bias in the model update
        return raw_grad + torch.ones_like(raw_grad) * NUM_CLIENTS
        
    elif attack_type == 'adaptive_attack':
        # Adds random noise to the gradient to evade detection mechanisms
        # The standard deviation of 0.1 makes the attack more subtle
        # For a more sophisticated adaptive attack, we could optimize the noise based on the defense
        noise_level = 0.1 * torch.norm(raw_grad) / torch.sqrt(torch.tensor(raw_grad.numel()).float())
        return raw_grad + noise_level * torch.randn_like(raw_grad)
        
    elif attack_type == 'min_max_attack':
        # Min-max attack as described in FLTrust paper
        # Attempts to maximize the negative impact on benign clients while minimizing detection chance
        # First normalize the gradient to unit norm to avoid magnitude-based detection
        normalized_grad = raw_grad / (torch.norm(raw_grad) + 1e-8)
        # Scale the normalized gradient to maximize impact
        return -normalized_grad * NUM_CLIENTS
        
    elif attack_type == 'min_sum_attack':
        # Min-sum attack as described in FLTrust paper
        # Focuses on minimizing the sum of cosine similarities with benign updates
        # The negative gradient with unit norm is a simple approach to this
        normalized_grad = raw_grad / (torch.norm(raw_grad) + 1e-8)
        return -normalized_grad * NUM_CLIENTS
        
    elif attack_type == 'alternating_attack':
        # Alternating attack - oscillating between positive and negative values
        # This makes the gradient difficult to detect by traditional methods
        sign_mask = torch.ones_like(raw_grad)
        # Create an alternating pattern of +1 and -1
        for i in range(sign_mask.numel()):
            idx = torch.tensor(i).reshape(1, -1)
            if i % 2 == 0:
                sign_mask.view(-1)[i] = 1
            else:
                sign_mask.view(-1)[i] = -1
        return raw_grad * sign_mask * NUM_CLIENTS
        
    elif attack_type == 'targeted_attack':
        # Targeted attack aims to introduce a specific pattern into the model
        # It focuses on a subset of parameters that affect a particular task/class
        # Here we simulate by targeting specific parts of the gradient (e.g., first 10%)
        mask = torch.zeros_like(raw_grad)
        # Target first 10% of parameters
        num_elements = int(0.1 * raw_grad.numel())
        mask.view(-1)[:num_elements] = 1
        return raw_grad * (1 - mask) + mask * NUM_CLIENTS * torch.randn_like(raw_grad)
        
    elif attack_type == 'gradient_inversion_attack':
        # Gradient inversion attack with different magnitudes for different parts
        # Creates an oscillating pattern with varying amplitudes
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
        
        return modified_grad
        
    else:
        # No attack
        return raw_grad 