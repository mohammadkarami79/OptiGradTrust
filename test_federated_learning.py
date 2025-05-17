import torch
import os
import numpy as np
from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset import load_dataset
from federated_learning.utils.model_utils import set_random_seeds
import traceback

def test_federated_learning_flow():
    """Test the complete federated learning flow with dual attention."""
    print("\n=== Testing Federated Learning Flow with Dual Attention ===")
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Set random seed for reproducibility
        set_random_seeds(42)
        
        # Load dataset
        print("Loading dataset...")
        train_dataset, test_dataset, num_classes, _ = load_dataset()
        print(f"Loaded dataset with {len(train_dataset)} samples")
        
        # Create data loaders with smaller subsets for quick testing
        print("Creating data loaders...")
        
        # Use much smaller datasets for quick testing
        train_size = min(5000, len(train_dataset))
        test_size = min(1000, len(test_dataset))
        
        train_indices = np.random.choice(len(train_dataset), size=train_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), size=test_size, replace=False)
        
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Split train dataset into root and client datasets
        root_size = int(0.2 * len(train_subset))  # 20% for root
        client_size = len(train_subset) - root_size
        
        all_indices = list(range(len(train_subset)))
        root_indices = np.random.choice(all_indices, size=root_size, replace=False)
        client_indices = np.array([i for i in all_indices if i not in root_indices])
        
        # Create root dataset
        root_dataset = torch.utils.data.Subset(train_subset, root_indices)
        root_loader = torch.utils.data.DataLoader(
            root_dataset, batch_size=32, shuffle=True,
            num_workers=0, pin_memory=False
        )
        
        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=32, shuffle=False,
            num_workers=0, pin_memory=False
        )
        
        # Create client datasets (use fewer clients for testing)
        num_clients = 3  # Reduced from default
        num_malicious = 1  # One malicious client
        
        num_samples_per_client = client_size // num_clients
        client_datasets = []
        
        for i in range(num_clients):
            start_idx = i * num_samples_per_client
            end_idx = (i+1) * num_samples_per_client if i < num_clients - 1 else len(client_indices)
            client_idx = client_indices[start_idx:end_idx]
            client_datasets.append(torch.utils.data.Subset(train_subset, client_idx))
        
        # Create server
        print("Creating server...")
        server = Server()
        server.set_datasets(root_loader, test_subset)
        
        # Create clients
        print(f"Creating {num_clients} clients with {num_malicious} malicious...")
        clients = []
        malicious_indices = np.random.choice(num_clients, size=num_malicious, replace=False)
        
        for i in range(num_clients):
            is_malicious = i in malicious_indices
            clients.append(Client(client_id=i, dataset=client_datasets[i], is_malicious=is_malicious))
            print(f"Client {i} created with {len(client_datasets[i])} samples (Malicious: {is_malicious})")
        
        # Assign clients to server
        server.clients = clients
        print(f"Assigned {len(clients)} clients to server")
        
        # Pretrain global model on root dataset (reduced epochs)
        print("\n--- Pretraining global model on root dataset ---")
        # Override global setting to use fewer epochs
        LOCAL_EPOCHS_ROOT = 1
        server._pretrain_global_model()
        
        # Collect root gradients for VAE training
        print("\n--- Collecting root gradients for VAE training ---")
        root_gradients = server._collect_root_gradients()
        print(f"Collected {len(root_gradients)} root gradients")
        
        # Train VAE on root gradients (reduced epochs)
        print("\n--- Training VAE on root gradients ---")
        server.vae = server.train_vae(root_gradients, vae_epochs=1)
        
        # Store root gradients for reference
        server.root_gradients = root_gradients
        
        # Test server feature computation
        print("\n--- Testing feature computation ---")
        if len(root_gradients) > 0:
            features = server._compute_gradient_features(root_gradients[0], root_gradients[0])
            print(f"Feature vector shape: {features.shape}")
            print(f"Feature values: {features}")
        
        # Train dual attention model (use synthetic data to avoid long client training)
        print("\n--- Training dual attention model ---")
        
        # Generate honest features from root gradients
        honest_features = torch.zeros((min(5, len(root_gradients)), 6 if ENABLE_SHAPLEY else 5), device=device)
        for i in range(min(5, len(root_gradients))):
            honest_features[i] = server._compute_gradient_features(root_gradients[i], root_gradients[0])
        
        # Generate synthetic malicious features
        malicious_features = server._generate_malicious_features(honest_features)
        
        # Train dual attention model with just 5 epochs
        from federated_learning.training.training_utils import train_dual_attention
        server.dual_attention = train_dual_attention(
            honest_features=honest_features,
            malicious_features=malicious_features,
            epochs=5,
            batch_size=8,
            lr=0.001
        )
        
        # Run federated learning for just 1 round
        print("\n--- Running federated learning (1 round) ---")
        test_errors, round_metrics = server.train(num_rounds=1)
        
        # Check results
        print("\n--- Final Results ---")
        print(f"Test errors: {test_errors}")
        
        print("\n=== Federated Learning Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"Error during federated learning test: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_federated_learning_flow() 