import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import os
from torchvision import datasets, transforms

# Import federated learning components
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.config.config import *
from federated_learning.models.vae import GradientVAE

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")

def load_mnist_data():
    """Load MNIST dataset and split for federated learning."""
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    train_dataset = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform)
    
    # Create a small subset for testing
    train_size = 6000  # Small subset for faster testing
    test_size = 1000
    
    # Create a random subset of training data
    indices = torch.randperm(len(train_dataset))
    train_indices = indices[:train_size]
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    
    # Create a random subset of test data
    test_indices = torch.randperm(len(test_dataset))[:test_size]
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create root dataset (a small part of training data)
    root_size = int(0.1 * train_size)  # 10% of training data
    root_indices = train_indices[:root_size]
    root_dataset = torch.utils.data.Subset(train_dataset, root_indices)
    
    # Create client datasets
    num_clients = 5
    samples_per_client = (train_size - root_size) // num_clients
    client_datasets = []
    
    # Create client data splits (non-IID to simulate realistic scenario)
    # We'll sort by label for the first 2 clients to create a non-IID split
    # Client 1: Mainly digits 0-4
    # Client 2: Mainly digits 5-9
    # Client 3-5: Random distribution
    
    # Get labels for remaining training data
    remaining_indices = train_indices[root_size:]
    remaining_labels = [train_dataset.targets[i].item() for i in remaining_indices]
    
    # Sort indices by label
    indices_by_label = [[] for _ in range(10)]
    for idx, label in zip(remaining_indices, remaining_labels):
        indices_by_label[label].append(idx)
    
    # Create client datasets
    client_indices = [[] for _ in range(num_clients)]
    
    # Client 1: Mainly digits 0-4
    for digit in range(5):
        # Get 80% of the digit's data for client 1
        digit_size = len(indices_by_label[digit])
        client_1_size = int(0.8 * digit_size)
        client_indices[0].extend(indices_by_label[digit][:client_1_size])
        # Move used indices to the end of the list
        indices_by_label[digit] = indices_by_label[digit][client_1_size:] + indices_by_label[digit][:client_1_size]
    
    # Client 2: Mainly digits 5-9
    for digit in range(5, 10):
        # Get 80% of the digit's data for client 2
        digit_size = len(indices_by_label[digit])
        client_2_size = int(0.8 * digit_size)
        client_indices[1].extend(indices_by_label[digit][:client_2_size])
        # Move used indices to the end of the list
        indices_by_label[digit] = indices_by_label[digit][client_2_size:] + indices_by_label[digit][:client_2_size]
    
    # Distribute remaining data randomly among all clients
    all_remaining = []
    for digit in range(10):
        all_remaining.extend(indices_by_label[digit])
    
    # Shuffle remaining data
    random.shuffle(all_remaining)
    
    # Distribute to clients 3-5 evenly and add some to clients 1-2 for balance
    samples_remaining = len(all_remaining)
    samples_per_client_remaining = samples_remaining // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client_remaining
        end_idx = start_idx + samples_per_client_remaining
        client_indices[i].extend(all_remaining[start_idx:end_idx])
    
    # Create Subset datasets for each client
    for i in range(num_clients):
        client_datasets.append(torch.utils.data.Subset(train_dataset, client_indices[i]))
    
    # Verify data distribution
    print(f"Total dataset size: {len(train_dataset)}")
    print(f"Root dataset size: {len(root_dataset)}")
    print(f"Test dataset size: {len(test_subset)}")
    
    for i, dataset in enumerate(client_datasets):
        print(f"Client {i} dataset size: {len(dataset)}")
    
    # Create DataLoaders
    root_loader = torch.utils.data.DataLoader(
        root_dataset, batch_size=32, shuffle=True, num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=32, shuffle=False, num_workers=0
    )
    
    return root_dataset, client_datasets, test_subset, root_loader, test_loader

def main():
    # Set random seeds for reproducibility
    set_seeds(42)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== GPU Configuration ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Display GPU memory stats
        print("\nGPU Memory:")
        print(f"Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    else:
        print("No GPU available, using CPU.")
    
    # Load datasets
    root_dataset, client_datasets, test_dataset, root_loader, test_loader = load_mnist_data()
    
    # Create server and clients
    print("\n--- Creating server and clients ---")
    server = Server()
    
    # Set datasets for server
    server.set_datasets(root_loader, test_dataset)
    
    # Create clients (make 2 of them malicious)
    clients = []
    for i in range(len(client_datasets)):
        # Make clients 1 and 2 malicious
        is_malicious = i in [1, 2]
        client = Client(i, client_datasets[i], is_malicious=is_malicious)
        clients.append(client)
        print(f"Created Client {i} (Malicious: {is_malicious})")
    
    # Add clients to server
    server.add_clients(clients)
    
    # Pretrain global model on root dataset
    print("\n--- Pretraining global model ---")
    server._pretrain_global_model()
    
    # Collect root gradients for VAE training
    print("\n--- Collecting root gradients for VAE training ---")
    server.root_gradients = server._collect_root_gradients()
    
    # Train VAE on root gradients
    print("\n--- Training VAE on root gradients ---")
    server.vae = server.train_vae(server.root_gradients, vae_epochs=2)
    
    # Run federated learning for a few rounds
    print("\n=== Starting Federated Learning ===")
    num_rounds = 3  # Small number of rounds for testing
    server.train(num_rounds=num_rounds)

if __name__ == "__main__":
    main() 