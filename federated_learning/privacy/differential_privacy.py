"""
Differential Privacy implementation for federated learning.

This module implements DP-SGD (Differentially Private Stochastic Gradient Descent)
based on the paper:
"Deep Learning with Differential Privacy" by Abadi et al.
"""

import torch
import math
import numpy as np
from federated_learning.config.config import DP_EPSILON, DP_DELTA, DP_CLIP_NORM

def clip_gradients(gradient, clip_norm):
    """
    Clip gradients to have a maximum L2 norm of clip_norm.
    
    Args:
        gradient (torch.Tensor): The gradient to be clipped
        clip_norm (float): Maximum L2 norm
        
    Returns:
        torch.Tensor: Clipped gradient
    """
    # Calculate current L2 norm
    current_norm = torch.norm(gradient)
    
    # Skip if gradient is already within bounds
    if current_norm <= clip_norm:
        return gradient
    
    # Calculate scaling factor and apply it
    scale = clip_norm / (current_norm + 1e-8)
    return gradient * scale

def add_noise(gradient, noise_scale, seed=None):
    """
    Add Gaussian noise to the gradient for differential privacy.
    
    Args:
        gradient (torch.Tensor): The gradient tensor
        noise_scale (float): Scale of the noise to add
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        torch.Tensor: Gradient with added noise
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate Gaussian noise with appropriate scale
    noise = torch.randn_like(gradient) * noise_scale
    
    # Add noise to gradient
    return gradient + noise

def calculate_noise_scale(epsilon, delta, clip_norm, batch_size, num_iterations=1):
    """
    Calculate the noise scale for Gaussian mechanism to satisfy (ε,δ)-DP.
    
    Args:
        epsilon (float): Privacy budget ε
        delta (float): Privacy leakage probability δ
        clip_norm (float): Gradient clipping norm
        batch_size (int): Batch size used in training
        num_iterations (int): Number of DP-SGD iterations
        
    Returns:
        float: Noise scale (σ) for the Gaussian mechanism
    """
    # Use analytical calibration of the Gaussian mechanism
    # This is a simplified calculation. More advanced approaches use 
    # the moments accountant method from the DP-SGD paper
    
    # Compute c² value
    c_squared = 2 * math.log(1.25 / delta)
    
    # Compute sampling probability (we divide by batch_size to approximate)
    sampling_prob = 1.0 / batch_size
    
    # Calculate noise multiplier
    noise_multiplier = (sampling_prob * math.sqrt(num_iterations * c_squared)) / epsilon
    
    # Calculate final noise scale
    noise_scale = clip_norm * noise_multiplier
    
    return noise_scale

def apply_differential_privacy(gradient, epsilon=None, delta=None, clip_norm=None, batch_size=32):
    """
    Apply differential privacy to a gradient using DP-SGD.
    
    Args:
        gradient (torch.Tensor): The gradient to be processed
        epsilon (float, optional): Privacy budget ε. If None, uses value from config.
        delta (float, optional): Privacy leakage probability δ. If None, uses value from config.
        clip_norm (float, optional): Gradient clipping norm. If None, uses value from config.
        batch_size (int, optional): Batch size used in training. Default is 32.
        
    Returns:
        torch.Tensor: Gradient with differential privacy applied
    """
    # Use config values if not provided
    if epsilon is None:
        epsilon = DP_EPSILON
    if delta is None:
        delta = DP_DELTA
    if clip_norm is None:
        clip_norm = DP_CLIP_NORM
    
    # Log DP parameters
    print(f"Applying Differential Privacy: ε={epsilon}, δ={delta}, clip norm={clip_norm}")
    
    # Step 1: Clip gradient
    clipped_gradient = clip_gradients(gradient, clip_norm)
    
    # Step 2: Calculate noise scale
    noise_scale = calculate_noise_scale(epsilon, delta, clip_norm, batch_size)
    print(f"Using noise scale σ={noise_scale:.6f}")
    
    # Step 3: Add calibrated noise
    dp_gradient = add_noise(clipped_gradient, noise_scale)
    
    return dp_gradient 