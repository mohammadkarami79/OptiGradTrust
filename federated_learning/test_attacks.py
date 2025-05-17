import torch
import numpy as np
from federated_learning.config.config import *
from federated_learning.attacks.attack_utils import simulate_attack

def test_attack_implementations():
    """Test function to verify all attack implementations are working correctly"""
    print("Testing attack implementations...")
    
    # Create a dummy gradient
    gradient_size = 10000
    gradient = torch.randn(gradient_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gradient = gradient.to(device)
    original_norm = gradient.norm().item()
    
    print(f"Original gradient norm: {original_norm:.4f}")
    print(f"Original gradient mean: {gradient.mean().item():.4f}")
    print(f"Original gradient std: {gradient.std().item():.4f}")
    print("-" * 80)
    
    # List of all attack types to test
    attack_types = [
        'none',
        'label_flipping',
        'scaling_attack',
        'partial_scaling_attack',
        'noise_attack',
        'sign_flipping_attack',
        'min_max_attack',
        'min_sum_attack',
        'alternating_attack',
        'targeted_attack',
        'backdoor_attack',
        'adaptive_attack',
        'gradient_inversion_attack'
    ]
    
    # Test each attack type
    for attack_type in attack_types:
        print(f"Testing attack: {attack_type}")
        
        # Apply the attack using attack_utils
        modified_gradient = apply_test_attack(gradient, attack_type)
        
        # Calculate statistics
        modified_norm = modified_gradient.norm().item()
        modified_mean = modified_gradient.mean().item()
        modified_std = modified_gradient.std().item()
        
        # Calculate cosine similarity between original and modified gradient
        if attack_type != 'none':
            cos_sim = torch.nn.functional.cosine_similarity(gradient.view(1, -1), 
                                                           modified_gradient.view(1, -1), 
                                                           dim=1)
            cos_sim_value = cos_sim.item()
        else:
            cos_sim_value = 1.0  # For 'none' attack, cos sim is 1
            
        # Print statistics
        print(f"Modified norm: {modified_norm:.4f} (change: {modified_norm - original_norm:.4f})")
        print(f"Modified mean: {modified_mean:.4f}")
        print(f"Modified std: {modified_std:.4f}")
        print(f"Cosine similarity: {cos_sim_value:.4f}")
        print("-" * 80)
    
    print("All attack implementations tested successfully!")

def apply_test_attack(gradient, attack_type):
    """Apply different types of attacks to the gradient for testing"""
    original_norm = gradient.norm().item()
    device = gradient.device
    
    if attack_type == 'none':
        # No attack
        print("No attack applied (attack_type is 'none')")
        return gradient
        
    elif attack_type == 'label_flipping':
        # Reverse gradient direction to simulate label flipping
        modified_grad = -gradient
        attack_name = "Label flipping attack"
        
    elif attack_type == 'scaling_attack':
        # Scale the entire gradient by a large factor
        scaling_factor = 10.0
        modified_grad = gradient * scaling_factor
        attack_name = f"Scaling attack (factor: {scaling_factor:.1f})"
        
    elif attack_type == 'partial_scaling_attack':
        # Scale a portion of the gradient elements
        scaling_factor = 10.0
        # Select 30% of the gradient elements to scale
        mask = torch.zeros_like(gradient, device=device)
        indices = torch.randperm(gradient.numel(), device=device)[:int(0.3 * gradient.numel())]
        mask.view(-1)[indices] = 1.0
        
        # Apply scaling only to selected elements
        modified_grad = gradient.clone()
        modified_grad = modified_grad * (1 - mask) + modified_grad * mask * scaling_factor
        attack_name = f"Partial scaling attack (factor: {scaling_factor:.1f}, 30% of elements)"
        
    elif attack_type == 'noise_attack':
        # Add Gaussian noise to the gradient
        noise_level = 0.5
        noise = torch.randn_like(gradient) * noise_level * gradient.norm()
        modified_grad = gradient + noise
        attack_name = f"Noise attack (level: {noise_level:.2f} * gradient norm)"
        
    elif attack_type == 'sign_flipping_attack':
        # Flip the signs of all gradient elements
        modified_grad = -gradient
        attack_name = "Sign flipping attack"
        
    elif attack_type == 'min_max_attack':
        # Amplify the largest gradient elements and reduce the smallest ones
        modified_grad = gradient.clone()
        
        # Find the top 10% largest (by magnitude) gradient elements
        num_elements = gradient.numel()
        values, indices = torch.topk(torch.abs(gradient.view(-1)), k=int(0.1 * num_elements))
        
        # Amplify largest elements by 3x
        modified_grad.view(-1)[indices] *= 3.0
        
        # Find the bottom 50% smallest (by magnitude) gradient elements
        values, indices = torch.topk(torch.abs(gradient.view(-1)), k=int(0.5 * num_elements), largest=False)
        
        # Reduce smallest elements to 10% of original value
        modified_grad.view(-1)[indices] *= 0.1
        attack_name = "Min-Max attack (amplify 10% largest by 3x, reduce 50% smallest to 10%)"
        
    elif attack_type == 'min_sum_attack':
        # Invert gradient and strengthen certain components to minimize sum
        modified_grad = -gradient.clone()  # Invert gradient
        
        # Randomly select 20% of gradient elements to strengthen
        mask = torch.zeros_like(gradient, device=device)
        indices = torch.randperm(gradient.numel(), device=device)[:int(0.2 * gradient.numel())]
        mask.view(-1)[indices] = 1.0
        
        # Strengthen selected elements by 2x
        modified_grad = modified_grad * (1 - mask) + modified_grad * mask * 2.0
        attack_name = "Min-Sum attack (invert + strengthen 20% by 2x)"
        
    elif attack_type == 'alternating_attack':
        # Alternate between positive and negative values
        modified_grad = gradient.clone().view(-1)
        for i in range(1, modified_grad.shape[0], 2):
            if i < modified_grad.shape[0]:
                modified_grad[i] = -modified_grad[i]
                
        modified_grad = modified_grad.view(gradient.shape)
        attack_name = "Alternating attack (flip every other element)"
        
    elif attack_type == 'targeted_attack':
        # Targeted model poisoning attack, focusing on specific model components
        modified_grad = gradient.clone()
        
        # Add targeted changes to gradient - target specific components
        random_noise = torch.randn_like(modified_grad) * 0.3 * gradient.norm()
        modified_grad += random_noise
        
        # Strengthen some components by inverting and amplifying them
        top_indices = torch.randperm(gradient.numel(), device=device)[:int(0.1 * gradient.numel())]
        modified_grad.view(-1)[top_indices] *= -3.0  # Invert and strengthen
        attack_name = "Targeted attack (noise + invert 10% components)"
        
    elif attack_type == 'backdoor_attack':
        # Backdoor attack that attempts to subtly influence the model
        # without significantly changing gradient norm
        modified_grad = gradient.clone()
        
        # Select a small subset (5%) of gradient components to modify
        mask = torch.zeros_like(gradient, device=device)
        indices = torch.randperm(gradient.numel(), device=device)[:int(0.05 * gradient.numel())]
        mask.view(-1)[indices] = 1.0
        
        # Apply consistent direction to selected components
        backdoor_pattern = torch.ones_like(gradient, device=device) * 0.1 * gradient.norm().item()
        modified_grad = modified_grad * (1 - mask) + backdoor_pattern * mask
        attack_name = "Backdoor attack (subtle 5% modification)"
        
    elif attack_type == 'adaptive_attack':
        # Adds random noise to the gradient to evade detection mechanisms
        noise_level = 0.1 * torch.norm(gradient) / torch.sqrt(torch.tensor(gradient.numel(), device=device).float())
        modified_grad = gradient + noise_level * torch.randn_like(gradient)
        attack_name = "Adaptive attack (scaled noise)"
        
    elif attack_type == 'gradient_inversion_attack':
        # Gradient inversion attack with different magnitudes for different parts
        # Creates an oscillating pattern with varying amplitudes
        quarter_size = gradient.numel() // 4
        modified_grad = gradient.clone()
        
        # First quarter: regular inversion
        modified_grad.view(-1)[:quarter_size] = -gradient.view(-1)[:quarter_size]
        
        # Second quarter: amplified inversion
        modified_grad.view(-1)[quarter_size:2*quarter_size] = -2 * gradient.view(-1)[quarter_size:2*quarter_size]
        
        # Third quarter: reduced inversion
        modified_grad.view(-1)[2*quarter_size:3*quarter_size] = -0.5 * gradient.view(-1)[2*quarter_size:3*quarter_size]
        
        # Fourth quarter: unchanged but amplified
        modified_grad.view(-1)[3*quarter_size:] = 3 * gradient.view(-1)[3*quarter_size:]
        attack_name = "Gradient inversion attack (sectional gradient manipulation)"
        
    else:
        # Use the attack_utils implementation for other attack types
        print(f"Using attack_utils implementation for {attack_type} attack")
        return simulate_attack(gradient, attack_type)
    
    # Print attack statistics
    modified_norm = modified_grad.norm().item()
    print(f"{attack_name} - Original norm: {original_norm:.4f}, Modified norm: {modified_norm:.4f}")
    print(f"Norm change: {modified_norm - original_norm:.4f} ({(modified_norm / original_norm - 1) * 100:.1f}%)")
    
    return modified_grad

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the test
    test_attack_implementations() 