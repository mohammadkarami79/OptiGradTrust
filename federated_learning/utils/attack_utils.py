import torch
import numpy as np
import random
from typing import Any, Callable, Dict, List, Optional

class AttackBase:
    """Base class for attacks on client updates."""
    
    def __init__(self, client=None):
        self.client = client
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply the attack to the gradient.
        
        Args:
            gradient (torch.Tensor): The original gradient
            
        Returns:
            torch.Tensor: The modified gradient after attack
        """
        raise NotImplementedError("Attack method not implemented")
    
    def apply_gradient_attack(self, gradient: torch.Tensor) -> torch.Tensor:
        """Compatibility method for client code that expects this interface.
        
        Args:
            gradient (torch.Tensor): The original gradient
            
        Returns:
            torch.Tensor: The modified gradient after attack
        """
        return self.apply(gradient)


class ScalingAttack(AttackBase):
    """Scale the gradient by a factor to disrupt training."""
    
    def __init__(self, client=None, scale_factor: float = 10.0):
        super().__init__(client)
        self.scale_factor = scale_factor
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """Scale the entire gradient by the scale factor."""
        return gradient * self.scale_factor


class PartialScalingAttack(AttackBase):
    """Scale only a portion of the gradient."""
    
    def __init__(self, client=None, scale_factor: float = 10.0, fraction: float = 0.5):
        super().__init__(client)
        self.scale_factor = scale_factor
        self.fraction = fraction
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """Scale a random portion of the gradient."""
        attacked_gradient = gradient.clone()
        
        # Calculate how many elements to scale
        num_elements = gradient.numel()
        num_to_scale = int(num_elements * self.fraction)
        
        # Get random indices to scale
        flat_indices = torch.randperm(num_elements)[:num_to_scale]
        
        # Flatten the gradient, scale the selected elements, then reshape back
        flat_gradient = attacked_gradient.view(-1)
        flat_gradient[flat_indices] *= self.scale_factor
        
        return attacked_gradient


class SignFlippingAttack(AttackBase):
    """Flip the sign of the gradient."""
    
    def __init__(self, client=None):
        super().__init__(client)
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """Flip the sign of the entire gradient."""
        return -gradient


class NoiseAttack(AttackBase):
    """Add noise to the gradient."""
    
    def __init__(self, client=None, noise_factor: float = 1.0):
        super().__init__(client)
        self.noise_factor = noise_factor
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the gradient."""
        # Calculate standard deviation of the gradient
        grad_std = torch.std(gradient)
        
        # Generate noise with same shape as gradient
        noise = torch.randn_like(gradient) * grad_std * self.noise_factor
        
        return gradient + noise


class TargetedParametersAttack(AttackBase):
    """Target specific important parameters for modification."""
    
    def __init__(self, client=None, scale_factor: float = 10.0, target_percentage: float = 0.2):
        super().__init__(client)
        self.scale_factor = scale_factor
        self.target_percentage = target_percentage
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """Target parameters with the largest magnitude."""
        attacked_gradient = gradient.clone()
        
        # Flatten the gradient
        flat_gradient = attacked_gradient.view(-1)
        
        # Get indices of elements with largest absolute value
        num_elements = flat_gradient.numel()
        num_to_target = int(num_elements * self.target_percentage)
        
        # Get indices of largest magnitude elements
        _, indices = torch.topk(torch.abs(flat_gradient), num_to_target)
        
        # Modify those elements (scale up and flip sign)
        flat_gradient[indices] = -flat_gradient[indices] * self.scale_factor
        
        return attacked_gradient


class MinMaxAttack(AttackBase):
    """Minimize loss for one class and maximize for others."""
    
    def __init__(self, client=None, target_class: int = 0):
        super().__init__(client)
        self.target_class = target_class
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """Flip sign of gradient to affect different classes differently."""
        # This is a simplified version of min-max attack
        # In a real implementation, we'd need to identify class-specific parts of the gradient
        return -gradient


class MinSumAttack(AttackBase):
    """Try to minimize the sum of parameters to disrupt training."""
    
    def __init__(self, client=None, scale_factor: float = 10.0):
        super().__init__(client)
        self.scale_factor = scale_factor
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """Set gradient to a large positive value to push parameters to be negative."""
        # Scale the gradient and ensure it's positive
        attacked_gradient = torch.abs(gradient) * self.scale_factor
        
        return attacked_gradient


class LabelFlippingAttack(AttackBase):
    """Flip labels during client training.
    
    Note: This attack is different as it modifies the client's behavior
    rather than just manipulating the gradient after training.
    """
    
    def __init__(self, client=None, flip_probability: float = 0.5):
        super().__init__(client)
        self.flip_probability = flip_probability
        
    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        This attack doesn't directly modify gradients
        Instead, the client should flip labels during training
        But for compatibility, we still provide this method
        """
        # Simply return the original gradient since the attack happens elsewhere
        return gradient
    
    def apply_to_client(self, client):
        """Apply label flipping to the client's training process."""
        # Store original forward pass
        original_forward = client.forward
        
        # Define new forward pass with label flipping
        def forward_with_flipped_labels(x, y):
            # Flip labels with probability self.flip_probability
            if random.random() < self.flip_probability:
                num_classes = 10  # Assuming 10 classes (e.g., MNIST, CIFAR-10)
                # Generate random labels different from original
                new_labels = torch.randint(0, num_classes, y.shape, device=y.device)
                # Ensure flipped labels are different from original
                mask = new_labels == y
                if mask.any():
                    new_labels[mask] = (new_labels[mask] + 1) % num_classes
                y = new_labels
            
            # Call original forward pass with potentially flipped labels
            return original_forward(x, y)
        
        # Replace client's forward method
        client.forward = forward_with_flipped_labels


def apply_attack(client, attack_type: str) -> None:
    """Apply the specified attack to a client.
    
    Args:
        client: The client to apply the attack to
        attack_type: Type of attack to apply
    """
    attack_mapping = {
        'scaling_attack': ScalingAttack(client, scale_factor=10.0),
        'partial_scaling_attack': PartialScalingAttack(client, scale_factor=5.0, fraction=0.3),
        'sign_flipping_attack': SignFlippingAttack(client),
        'noise_attack': NoiseAttack(client, noise_factor=2.0),
        'targeted_parameters': TargetedParametersAttack(client, scale_factor=5.0, target_percentage=0.1),
        'label_flipping': LabelFlippingAttack(client, flip_probability=0.5),
        'min_max': MinMaxAttack(client),
        'min_sum': MinSumAttack(client, scale_factor=3.0),
        'none': None
    }
    
    if attack_type not in attack_mapping:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    attack = attack_mapping[attack_type]
    
    if attack is None:
        return
    
    # Store the attack in the client
    client.attack = attack
    
    # Special case for label flipping which modifies the client's behavior
    if attack_type == 'label_flipping':
        attack.apply_to_client(client)
    
    # Set the client as malicious
    client.is_malicious = True
    
    print(f"Applied {attack_type} to client {client.client_id}") 