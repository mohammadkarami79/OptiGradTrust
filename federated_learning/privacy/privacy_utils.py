"""
Utility functions for privacy mechanisms in federated learning.
"""

import torch
from federated_learning.config.config import *

def apply_privacy_mechanism(gradient, mechanism=None):
    """
    Apply the specified privacy mechanism to the gradient.
    
    Args:
        gradient (torch.Tensor): The gradient to be processed
        mechanism (str, optional): The privacy mechanism to use. 
                                  If None, uses PRIVACY_MECHANISM from config.
                                  
    Returns:
        torch.Tensor: The processed gradient with privacy mechanism applied
    """
    if mechanism is None:
        mechanism = PRIVACY_MECHANISM
        
    if mechanism == 'none':
        # No privacy mechanism, return gradient as is
        return gradient
    elif mechanism == 'dp':
        # Import here to avoid circular imports
        from federated_learning.privacy.differential_privacy import apply_differential_privacy
        return apply_differential_privacy(gradient, DP_EPSILON, DP_DELTA, DP_CLIP_NORM)
    elif mechanism == 'paillier':
        # Import here to avoid circular imports
        from federated_learning.privacy.homomorphic_encryption import apply_paillier_encryption
        return apply_paillier_encryption(gradient)
    else:
        print(f"Warning: Unknown privacy mechanism '{mechanism}'. Using 'none' instead.")
        return gradient 