"""
Homomorphic Encryption implementation for federated learning using Paillier cryptosystem.

This implementation requires the Python 'phe' library. Install it with:
pip install phe
"""

import torch
import numpy as np
import time
import pickle
import os
from federated_learning.config.config import *

try:
    from phe import paillier
    HAS_PAILLIER = True
except ImportError:
    HAS_PAILLIER = False
    print("Warning: phe library not found. Install with 'pip install phe' to use Paillier encryption.")

# Global variables to store keys
public_key = None
private_key = None
key_path = os.path.join('model_weights', 'paillier_keys.pkl')

def initialize_paillier(key_length=2048, regenerate=False):
    """
    Initialize Paillier cryptosystem by generating or loading keys.
    
    Args:
        key_length (int): Length of the key in bits (default: 2048)
        regenerate (bool): If True, regenerate keys even if they exist
        
    Returns:
        tuple: (public_key, private_key)
    """
    global public_key, private_key
    
    if not HAS_PAILLIER:
        print("Error: phe library is required for Paillier encryption")
        return None, None
    
    # Check if keys already exist
    if os.path.exists(key_path) and not regenerate:
        try:
            print(f"Loading existing Paillier keys from {key_path}")
            with open(key_path, 'rb') as f:
                keys = pickle.load(f)
                public_key = keys['public_key']
                private_key = keys['private_key']
        except Exception as e:
            print(f"Error loading keys: {e}. Regenerating...")
            public_key, private_key = None, None
    
    # Generate new keys if needed
    if public_key is None or private_key is None or regenerate:
        print(f"Generating new Paillier keys with length {key_length} bits...")
        start_time = time.time()
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
        end_time = time.time()
        print(f"Key generation completed in {end_time - start_time:.2f} seconds")
        
        # Save keys
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, 'wb') as f:
            pickle.dump({'public_key': public_key, 'private_key': private_key}, f)
        print(f"Paillier keys saved to {key_path}")
    
    return public_key, private_key

def encrypt_gradient(gradient, public_key):
    """
    Encrypt a gradient tensor using Paillier encryption.
    
    Args:
        gradient (torch.Tensor): The gradient tensor to encrypt
        public_key: Paillier public key
        
    Returns:
        list: List of encrypted values
    """
    # Convert gradient tensor to numpy array and flatten
    grad_np = gradient.detach().cpu().numpy().flatten()
    
    # Scale values to integers for better precision with Paillier
    # We use a fixed scaling factor and will rescale after decryption
    scaling_factor = 1e6
    grad_scaled = (grad_np * scaling_factor).astype(np.int64)
    
    # Encrypt each value
    print(f"Encrypting gradient with {len(grad_scaled)} elements...")
    start_time = time.time()
    encrypted_values = [public_key.encrypt(int(val)) for val in grad_scaled]
    end_time = time.time()
    
    # Store shape and scaling factor for decryption
    metadata = {
        'shape': gradient.shape,
        'scaling_factor': scaling_factor,
        'original_dtype': str(gradient.dtype)
    }
    
    print(f"Encryption completed in {end_time - start_time:.2f} seconds")
    return encrypted_values, metadata

def decrypt_gradient(encrypted_values, metadata, private_key):
    """
    Decrypt an encrypted gradient.
    
    Args:
        encrypted_values (list): List of encrypted values
        metadata (dict): Metadata about the original gradient
        private_key: Paillier private key
        
    Returns:
        torch.Tensor: Decrypted gradient tensor
    """
    print(f"Decrypting gradient with {len(encrypted_values)} elements...")
    start_time = time.time()
    
    # Decrypt values
    decrypted_values = [private_key.decrypt(val) for val in encrypted_values]
    
    # Rescale values back to original range
    scaling_factor = metadata['scaling_factor']
    decrypted_scaled = np.array(decrypted_values) / scaling_factor
    
    # Reshape to original shape
    decrypted_np = decrypted_scaled.reshape(metadata['shape'])
    
    # Convert back to torch tensor
    original_dtype = getattr(torch, metadata['original_dtype'])
    decrypted_tensor = torch.tensor(decrypted_np, dtype=original_dtype)
    
    end_time = time.time()
    print(f"Decryption completed in {end_time - start_time:.2f} seconds")
    
    return decrypted_tensor

def apply_paillier_encryption(gradient):
    """
    Apply Paillier homomorphic encryption to gradient, then immediately decrypt it.
    This simulates the process of encrypting for transmission, then decrypting for aggregation.
    
    In a real deployment, encryption would happen on the client side,
    and decryption would happen on the server side after aggregation.
    
    Args:
        gradient (torch.Tensor): The gradient to be processed
        
    Returns:
        torch.Tensor: The processed gradient (decrypted after encryption)
    """
    global public_key, private_key
    
    if not HAS_PAILLIER:
        print("Error: phe library is required for Paillier encryption")
        return gradient
    
    # Initialize keys if not already done
    if public_key is None or private_key is None:
        public_key, private_key = initialize_paillier()
        if public_key is None:
            print("Failed to initialize Paillier keys. Returning original gradient.")
            return gradient
    
    # Apply encryption
    try:
        encrypted_values, metadata = encrypt_gradient(gradient, public_key)
        
        # Decrypt the encrypted values
        # In real deployment, this would happen after transmission and aggregation
        decrypted_gradient = decrypt_gradient(encrypted_values, metadata, private_key)
        
        return decrypted_gradient
    except Exception as e:
        print(f"Error in Paillier encryption/decryption: {e}")
        print("Returning original gradient.")
        return gradient 