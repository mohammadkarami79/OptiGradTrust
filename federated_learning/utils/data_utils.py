import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import os
import random
from typing import Tuple, List, Dict, Optional, Union
from collections import defaultdict

from federated_learning.config.config import *


def get_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """Load dataset based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset to load ('MNIST', 'CIFAR10', 'ALZHEIMER')
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name == 'MNIST':
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load the training data
        train_dataset = torchvision.datasets.MNIST(
            root='./data/MNIST', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Download and load the test data
        test_dataset = torchvision.datasets.MNIST(
            root='./data/MNIST', 
            train=False, 
            download=True, 
            transform=transform
        )
        
    elif dataset_name == 'CIFAR10':
        # Define transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Download and load the training data
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar',
            train=True,
            download=True,
            transform=train_transform
        )
        
        # Download and load the test data
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar',
            train=False,
            download=True,
            transform=test_transform
        )
        
    elif dataset_name == 'ALZHEIMER':
        # Specific path for the Alzheimer dataset
        base_path = './data/alzheimer'
        
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Assuming the dataset is already structured in the right format
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(base_path, 'train'),
            transform=transform
        )
        
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(base_path, 'test'),
            transform=transform
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    return train_dataset, test_dataset


def create_federated_dataset(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int,
    root_dataset_ratio: float = 0.1,
    iid: bool = True,
    dirichlet_alpha: float = 1.0
) -> Tuple[List[Dataset], Dataset, DataLoader]:
    """Create federated datasets for clients.
    
    Args:
        train_dataset: The original training dataset
        test_dataset: The original test dataset
        num_clients: Number of clients to create datasets for
        root_dataset_ratio: Ratio of data to keep for the root dataset
        iid: Whether to distribute data in an IID fashion
        dirichlet_alpha: Concentration parameter for Dirichlet distribution (lower = more non-IID)
    
    Returns:
        Tuple of (client_datasets, root_dataset, test_loader)
    """
    # Extract the total number of training examples
    num_examples = len(train_dataset)
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Split the dataset to create the root dataset (trusted)
    num_root_examples = int(num_examples * root_dataset_ratio)
    num_client_examples = num_examples - num_root_examples
    
    root_dataset, client_dataset = random_split(
        train_dataset, 
        [num_root_examples, num_client_examples],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Create a DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS if NUM_WORKERS > 0 else 0,
        pin_memory=PIN_MEMORY
    )
    
    # Distribute remaining data to clients
    client_datasets = []
    
    # Get targets/labels from the dataset
    if hasattr(train_dataset, 'targets'):
        all_targets = train_dataset.targets
    elif isinstance(train_dataset, torchvision.datasets.ImageFolder):
        all_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    else:
        all_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    
    # Convert targets to numpy array if it's a tensor
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.numpy()
    
    # Get indices of client dataset portion
    client_indices = client_dataset.indices
    
    # Get the targets of the client dataset portion
    client_targets = [all_targets[i] for i in client_indices]
    
    # Determine number of classes
    if isinstance(all_targets, np.ndarray):
        num_classes = len(np.unique(all_targets))
    else:
        num_classes = len(set(all_targets))
    
    # Create client datasets
    if iid:
        # IID distribution: random split
        indices_per_client = np.array_split(client_indices, num_clients)
        client_datasets = [Subset(train_dataset, indices) for indices in indices_per_client]
    else:
        # Non-IID distribution using Dirichlet distribution
        class_indices = {}
        for class_id in range(num_classes):
            class_indices[class_id] = [
                idx for idx, target in zip(client_indices, client_targets) if target == class_id
            ]
        
        # Dirichlet distribution to allocate samples of each class to clients
        client_sample_indices = [[] for _ in range(num_clients)]
        
        for class_id in range(num_classes):
            num_samples_of_class = len(class_indices[class_id])
            if num_samples_of_class == 0:
                continue
                
            # Draw proportions from Dirichlet distribution
            proportions = np.random.dirichlet(
                np.repeat(dirichlet_alpha, num_clients)
            )
            
            # Calculate number of samples per client for this class
            num_samples_per_client = (proportions * num_samples_of_class).astype(int)
            
            # Adjust to make sure we allocate all samples
            while sum(num_samples_per_client) < num_samples_of_class:
                num_samples_per_client[np.random.randint(num_clients)] += 1
                
            # Allocate samples to clients
            class_indices_permuted = np.random.permutation(class_indices[class_id])
            start_idx = 0
            for client_id, num_samples in enumerate(num_samples_per_client):
                end_idx = start_idx + num_samples
                client_sample_indices[client_id].extend(
                    class_indices_permuted[start_idx:end_idx].tolist()
                )
                start_idx = end_idx
        
        # Create client datasets from indices
        client_datasets = [Subset(train_dataset, indices) for indices in client_sample_indices]
    
    # Calculate class distribution for each client (for logging)
    class_distributions = []
    for client_id, dataset in enumerate(client_datasets):
        indices = dataset.indices
        targets = [all_targets[i] for i in indices]
        
        # Count occurrences of each class
        class_counts = defaultdict(int)
        for target in targets:
            class_counts[int(target)] += 1
        
        # Convert to percentage
        total = sum(class_counts.values())
        class_percentages = {cls: (count / total) * 100 for cls, count in class_counts.items()}
        
        class_distributions.append(class_percentages)
    
    # Print data distribution information
    print(f"\n=== Data Distribution ===")
    print(f"Root dataset: {len(root_dataset)} samples")
    print(f"Client datasets: {[len(ds) for ds in client_datasets]}")
    print(f"Distribution type: {'IID' if iid else 'Non-IID'}")
    
    if not iid:
        print("\nClass distribution per client (%):")
        for client_id, dist in enumerate(class_distributions):
            print(f"Client {client_id}: ", end="")
            for cls, perc in sorted(dist.items()):
                print(f"Class {cls}: {perc:.1f}%  ", end="")
            print()
    
    return client_datasets, root_dataset, test_loader 