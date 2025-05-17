"""
Utility functions for dataset loading and dataset distribution for clients.
This module integrates with the existing dataset.py, providing higher-level functions
for dataset management in federated learning.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import random
import numpy as np
from federated_learning.config.config import *
from federated_learning.data.dataset import split_dataset, create_root_dataset

def load_dataset(dataset_name=None):
    """
    Load the dataset specified in config or by parameter.
    
    Args:
        dataset_name: Optional override for dataset name (not used in current implementation)
        
    Returns:
        train_dataset, test_dataset
    """
    # Import the original function
    from federated_learning.data.dataset import load_dataset as _load_base_dataset
    
    # Call the base dataset loader
    train_dataset, test_dataset, _, _ = _load_base_dataset()
    
    return train_dataset, test_dataset

def create_client_datasets(train_dataset, num_clients=None, root_size=None, iid=False, alpha=None):
    """
    Split a dataset into root dataset and client datasets.
    
    Args:
        train_dataset: The full training dataset
        num_clients: Number of clients to create datasets for
        root_size: Size of the root dataset (percentage or absolute)
        iid: Whether to use IID distribution
        alpha: Dirichlet concentration parameter for non-IID
        
    Returns:
        root_dataset, list_of_client_datasets
    """
    # Use config values if not specified
    if num_clients is None:
        num_clients = NUM_CLIENTS
        
    if root_size is None:
        root_size = ROOT_DATASET_SIZE
        
    if alpha is None and not iid:
        alpha = DIRICHLET_ALPHA
    
    # Determine number of classes
    if DATASET == 'MNIST':
        num_classes = 10
    elif DATASET == 'CIFAR10':
        num_classes = 10
    elif DATASET == 'ALZHEIMER':
        num_classes = ALZHEIMER_CLASSES
    else:
        # Try to infer from dataset
        if hasattr(train_dataset, 'targets'):
            num_classes = len(set(train_dataset.targets))
        else:
            # Default: assume 10 classes
            num_classes = 10
            print(f"Warning: Could not determine number of classes. Using default: {num_classes}")
    
    # Create root dataset
    root_dataset = create_root_dataset(train_dataset, num_classes)
    print(f"Created root dataset with {len(root_dataset)} samples")
    
    # Create client datasets
    if iid:
        distribution_type = 'iid'
    else:
        distribution_type = 'dirichlet'
    
    # Import specific split function based on distribution type
    from federated_learning.data.dataset import (
        split_dataset_iid, 
        split_dataset_dirichlet
    )
    
    # Create list of client datasets
    if iid:
        client_datasets = split_dataset_iid(train_dataset, num_classes)
    else:
        # Use the dirichlet function directly to pass alpha
        client_datasets = split_dataset_dirichlet(train_dataset, num_classes, alpha)
    
    print(f"Created {len(client_datasets)} client datasets")
    
    return root_dataset, client_datasets

def apply_attack_to_dataset(dataset, attack_type, num_classes=10):
    """
    Apply an attack to a dataset.
    
    Args:
        dataset: The dataset to apply the attack to
        attack_type: Attack type (label_flipping, backdoor, etc.)
        num_classes: Number of classes in the dataset
        
    Returns:
        Poisoned dataset
    """
    from federated_learning.data.dataset import (
        LabelFlippingDataset, 
        BackdoorDataset,
        MinMaxAttackDataset,
        TargetedAttackDataset
    )
    
    if attack_type == 'label_flipping':
        return LabelFlippingDataset(dataset, num_classes)
    elif attack_type == 'backdoor':
        return BackdoorDataset(dataset, num_classes)
    elif attack_type == 'min_max':
        return MinMaxAttackDataset(dataset, num_classes)
    elif attack_type == 'targeted':
        return TargetedAttackDataset(dataset, num_classes)
    else:
        print(f"Warning: Unknown attack type '{attack_type}'. Using original dataset.")
        return dataset 