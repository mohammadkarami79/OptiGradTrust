#!/usr/bin/env python3
"""
🏷️ LABEL SKEW NON-IID UTILITIES
===============================

Utilities for creating Label Skew Non-IID data distributions
for federated learning scenarios.

Author: Research Team
Date: 30 December 2025
Purpose: Complete Non-IID implementation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from collections import defaultdict
import random

class LabelSkewDataset(Dataset):
    """Custom dataset for Label Skew Non-IID distribution"""
    
    def __init__(self, original_dataset, client_indices):
        self.original_dataset = original_dataset
        self.client_indices = client_indices
    
    def __len__(self):
        return len(self.client_indices)
    
    def __getitem__(self, idx):
        original_idx = self.client_indices[idx]
        return self.original_dataset[original_idx]

def create_label_skew_distribution(dataset, num_clients, skew_factor=0.8, seed=42):
    """
    Create Label Skew Non-IID distribution where each client has 
    imbalanced class distributions.
    
    Args:
        dataset: Original dataset
        num_clients: Number of clients
        skew_factor: Degree of skew (0.0 = IID, 1.0 = extreme skew)
        seed: Random seed for reproducibility
    
    Returns:
        List of client datasets
    """
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Group indices by labels
    class_indices = defaultdict(list)
    
    for idx in range(len(dataset)):
        try:
            # Handle different dataset formats
            if hasattr(dataset, 'targets'):
                label = dataset.targets[idx]
            elif hasattr(dataset, 'labels'):
                label = dataset.labels[idx]
            else:
                _, label = dataset[idx]
                
            if torch.is_tensor(label):
                label = label.item()
                
            class_indices[label].append(idx)
        except Exception as e:
            print(f"Warning: Could not get label for index {idx}: {e}")
            continue
    
    num_classes = len(class_indices)
    classes = list(class_indices.keys())
    
    print(f"📊 Label Skew Distribution:")
    print(f"   Classes: {num_classes}")
    print(f"   Clients: {num_clients}")
    print(f"   Skew factor: {skew_factor}")
    
    # Initialize client datasets
    client_datasets = []
    
    for client_id in range(num_clients):
        client_indices = []
        
        # Assign 1-2 dominant classes to each client
        dominant_classes = [
            classes[(client_id + i) % num_classes] 
            for i in range(2)
        ]
        
        for class_label in classes:
            class_data = class_indices[class_label]
            
            if class_label in dominant_classes:
                # Dominant class: higher proportion
                proportion = skew_factor / len(dominant_classes)
            else:
                # Minor classes: lower proportion
                num_minor_clients = max(1, num_clients - len(dominant_classes))
                proportion = (1 - skew_factor) / num_minor_clients
            
            # Calculate number of samples for this client
            num_samples = int(len(class_data) * proportion)
            
            # Random sampling with replacement if needed
            if num_samples > 0:
                if num_samples <= len(class_data):
                    selected_indices = np.random.choice(
                        class_data, 
                        size=num_samples, 
                        replace=False
                    )
                else:
                    selected_indices = np.random.choice(
                        class_data, 
                        size=num_samples, 
                        replace=True
                    )
                
                client_indices.extend(selected_indices.tolist())
        
        # Shuffle client indices
        np.random.shuffle(client_indices)
        
        # Create client dataset
        if len(client_indices) > 0:
            client_dataset = LabelSkewDataset(dataset, client_indices)
            client_datasets.append(client_dataset)
            
            # Print client statistics
            client_labels = []
            for idx in client_indices[:100]:  # Sample to avoid memory issues
                try:
                    if hasattr(dataset, 'targets'):
                        label = dataset.targets[idx]
                    else:
                        _, label = dataset[idx]
                    if torch.is_tensor(label):
                        label = label.item()
                    client_labels.append(label)
                except:
                    pass
            
            if client_labels:
                unique_labels, counts = np.unique(client_labels, return_counts=True)
                dominant_label = unique_labels[np.argmax(counts)]
                dominance = np.max(counts) / len(client_labels) * 100
                
                print(f"   Client {client_id}: {len(client_indices)} samples, "
                      f"dominant class {dominant_label} ({dominance:.1f}%)")
        else:
            print(f"   Warning: Client {client_id} has no data")
            # Create empty dataset
            client_dataset = LabelSkewDataset(dataset, [])
            client_datasets.append(client_dataset)
    
    return client_datasets

def analyze_label_skew_distribution(client_datasets, original_dataset):
    """
    Analyze the created Label Skew distribution
    
    Args:
        client_datasets: List of client datasets
        original_dataset: Original dataset for comparison
    
    Returns:
        Dictionary with analysis results
    """
    
    analysis = {
        'num_clients': len(client_datasets),
        'total_samples': sum(len(ds) for ds in client_datasets),
        'client_stats': [],
        'skew_metrics': {}
    }
    
    for client_id, client_dataset in enumerate(client_datasets):
        if len(client_dataset) == 0:
            continue
            
        # Get label distribution for this client
        client_labels = []
        for i in range(min(len(client_dataset), 1000)):  # Sample to avoid memory
            try:
                _, label = client_dataset[i]
                if torch.is_tensor(label):
                    label = label.item()
                client_labels.append(label)
            except:
                pass
        
        if client_labels:
            unique_labels, counts = np.unique(client_labels, return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            
            # Calculate skew metrics
            total_samples = sum(counts)
            proportions = counts / total_samples
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            max_proportion = np.max(proportions)
            
            client_stat = {
                'client_id': client_id,
                'num_samples': len(client_dataset),
                'num_classes': len(unique_labels),
                'label_distribution': label_dist,
                'entropy': entropy,
                'max_class_proportion': max_proportion,
                'skew_score': 1 - entropy / np.log2(len(unique_labels) + 1e-10)
            }
            
            analysis['client_stats'].append(client_stat)
    
    # Overall skew metrics
    if analysis['client_stats']:
        skew_scores = [stat['skew_score'] for stat in analysis['client_stats']]
        entropies = [stat['entropy'] for stat in analysis['client_stats']]
        
        analysis['skew_metrics'] = {
            'average_skew_score': np.mean(skew_scores),
            'std_skew_score': np.std(skew_scores),
            'average_entropy': np.mean(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies)
        }
    
    return analysis

def print_label_skew_summary(analysis):
    """Print a summary of Label Skew analysis"""
    
    print(f"\n📊 LABEL SKEW ANALYSIS SUMMARY")
    print("="*40)
    
    print(f"Total clients: {analysis['num_clients']}")
    print(f"Total samples: {analysis['total_samples']}")
    
    if analysis['skew_metrics']:
        metrics = analysis['skew_metrics']
        print(f"\n🔍 Skew Metrics:")
        print(f"   Average skew score: {metrics['average_skew_score']:.3f}")
        print(f"   Skew std deviation: {metrics['std_skew_score']:.3f}")
        print(f"   Average entropy: {metrics['average_entropy']:.3f}")
        print(f"   Entropy range: {metrics['min_entropy']:.3f} - {metrics['max_entropy']:.3f}")
    
    print(f"\n👥 Client Details:")
    for stat in analysis['client_stats'][:5]:  # Show first 5 clients
        print(f"   Client {stat['client_id']}: {stat['num_samples']} samples, "
              f"{stat['num_classes']} classes, skew: {stat['skew_score']:.3f}")
    
    if len(analysis['client_stats']) > 5:
        print(f"   ... and {len(analysis['client_stats']) - 5} more clients")

# Example usage function
def create_label_skew_example():
    """Example of how to use Label Skew utilities"""
    
    print(f"\n🛠️ LABEL SKEW IMPLEMENTATION EXAMPLE")
    print("="*40)
    
    example_code = '''
# Example usage in your federated learning setup:

from federated_learning.utils.label_skew_utils import create_label_skew_distribution

# Load your dataset (MNIST, CIFAR-10, etc.)
train_dataset = load_dataset()

# Create Label Skew Non-IID distribution
client_datasets = create_label_skew_distribution(
    dataset=train_dataset,
    num_clients=10,
    skew_factor=0.8,  # 80% skew towards dominant classes
    seed=42
)

# Use in your federated learning training
for client_id, client_dataset in enumerate(client_datasets):
    client = Client(client_id=client_id, dataset=client_dataset)
    # ... continue with federated training
    '''
    
    print(example_code)
    print("✅ Label Skew implementation complete!")

if __name__ == "__main__":
    create_label_skew_example() 