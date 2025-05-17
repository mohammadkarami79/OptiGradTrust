import torch
import random
import numpy as np
import os
import sys
import traceback
from datetime import datetime

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    from federated_learning.config.config import *
    from federated_learning.data.dataset import load_dataset, split_dataset, create_root_dataset
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    
    print("\n[SUCCESS] All imports completed successfully")
    
    def test_load_dataset():
        print("\n=== Step 1: Loading Dataset ===")
        try:
            train_dataset, test_dataset, num_classes, input_channels = load_dataset()
            print(f"Dataset loaded successfully")
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Test dataset size: {len(test_dataset)}")
            print(f"Number of classes: {num_classes}")
            print(f"Input channels: {input_channels}")
            return train_dataset, test_dataset, num_classes, input_channels
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            traceback.print_exc()
            return None, None, None, None
    
    def test_create_root_dataset(train_dataset):
        print("\n=== Step 2: Creating Root Dataset ===")
        try:
            root_dataset = create_root_dataset(train_dataset, num_classes=10)
            print(f"Root dataset created successfully with {len(root_dataset)} samples")
            
            # Create remaining dataset
            root_indices = set(root_dataset.indices)
            all_indices = set(range(len(train_dataset)))
            remaining_indices = list(all_indices - root_indices)
            print(f"Remaining dataset size: {len(remaining_indices)}")
            
            return root_dataset, remaining_indices
        except Exception as e:
            print(f"Error creating root dataset: {str(e)}")
            traceback.print_exc()
            return None, None
    
    def test_split_dataset(train_dataset, remaining_indices, num_classes):
        print("\n=== Step 3: Splitting Dataset Among Clients ===")
        try:
            client_datasets = split_dataset(
                torch.utils.data.Subset(train_dataset, remaining_indices),
                num_classes,
                distribution_type=DATA_DISTRIBUTION
            )
            print(f"Dataset split successfully among {len(client_datasets)} clients")
            
            for i, dataset in enumerate(client_datasets):
                print(f"Client {i} dataset size: {len(dataset)}")
            
            return client_datasets
        except Exception as e:
            print(f"Error splitting dataset: {str(e)}")
            traceback.print_exc()
            return None
    
    def test_initialize_server():
        print("\n=== Step 4: Initializing Server ===")
        try:
            server = Server()
            print("Server initialized successfully")
            print(f"Server has global model: {server.global_model is not None}")
            print(f"Server has VAE: {server.vae is not None}")
            print(f"Server has dual attention: {server.dual_attention is not None}")
            
            # Check device placement
            print(f"Global model device: {next(server.global_model.parameters()).device}")
            print(f"VAE device: {next(server.vae.parameters()).device}")
            print(f"Dual attention device: {next(server.dual_attention.parameters()).device}")
            
            return server
        except Exception as e:
            print(f"Error initializing server: {str(e)}")
            traceback.print_exc()
            return None
    
    def test_initialize_clients(client_datasets):
        print("\n=== Step 5: Initializing Clients ===")
        try:
            clients = []
            malicious_clients = []
            
            # Determine malicious client indices
            malicious_indices = random.sample(range(NUM_CLIENTS), NUM_MALICIOUS)
            print(f"Selected {len(malicious_indices)} clients as malicious: {malicious_indices}")
            
            for i in range(NUM_CLIENTS):
                is_malicious = i in malicious_indices
                
                # Create client
                client = Client(
                    client_id=i,
                    dataset=client_datasets[i],
                    is_malicious=is_malicious,
                    num_classes=10
                )
                
                # Add to client list
                clients.append(client)
                
                if is_malicious:
                    print(f"Client {i}: Initialized as malicious with {ATTACK_TYPE} attack")
                    malicious_clients.append(i)
            
            print(f"All {len(clients)} clients initialized successfully")
            return clients, malicious_clients
        except Exception as e:
            print(f"Error initializing clients: {str(e)}")
            traceback.print_exc()
            return None, None
    
    def test_setup_server(server, root_dataset, test_dataset, clients, malicious_clients):
        print("\n=== Step 6: Setting Up Server ===")
        try:
            # Create root dataset loader
            root_loader = torch.utils.data.DataLoader(
                root_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            
            # Set datasets on server
            server.set_datasets(root_loader, test_dataset)
            print("Datasets set on server successfully")
            
            # Set clients on server
            server.clients = clients
            server.malicious_clients = malicious_clients
            print("Clients set on server successfully")
            
            return server
        except Exception as e:
            print(f"Error setting up server: {str(e)}")
            traceback.print_exc()
            return None
    
    def test_pretrain_global_model(server):
        print("\n=== Step 7: Pre-training Global Model ===")
        try:
            server._pretrain_global_model()
            print("Global model pre-trained successfully")
            
            # Evaluate initial model
            test_loader = torch.utils.data.DataLoader(
                server.test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            from federated_learning.training.training_utils import test
            initial_acc, initial_error = test(server.global_model, test_loader)
            print(f"Initial test error after pretraining: {initial_error:.4f}")
            print(f"Initial test accuracy after pretraining: {initial_acc:.4f}")
            
            return True
        except Exception as e:
            print(f"Error pre-training global model: {str(e)}")
            traceback.print_exc()
            return False
    
    def main():
        # Set random seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print("\n=== Running Main Test Script ===")
        
        # Step 1: Load dataset
        train_dataset, test_dataset, num_classes, input_channels = test_load_dataset()
        if train_dataset is None:
            return
        
        # Step 2: Create root dataset
        root_dataset, remaining_indices = test_create_root_dataset(train_dataset)
        if root_dataset is None:
            return
        
        # Step 3: Split dataset among clients
        client_datasets = test_split_dataset(train_dataset, remaining_indices, num_classes)
        if client_datasets is None:
            return
        
        # Step 4: Initialize server
        server = test_initialize_server()
        if server is None:
            return
        
        # Step 5: Initialize clients
        clients, malicious_clients = test_initialize_clients(client_datasets)
        if clients is None:
            return
        
        # Step 6: Set up server with datasets and clients
        server = test_setup_server(server, root_dataset, test_dataset, clients, malicious_clients)
        if server is None:
            return
        
        # Step 7: Pre-train global model
        if not test_pretrain_global_model(server):
            return
        
        print("\n=== All Steps Completed Successfully ===")
    
    # Run the main function
    if __name__ == "__main__":
        main()
        
except Exception as e:
    print(f"Unhandled error: {str(e)}")
    traceback.print_exc() 