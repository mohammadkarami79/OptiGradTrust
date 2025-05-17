"""
Main script for running the complete federated learning system with dual attention
for malicious client detection.

This script performs the following steps:
1. Load dataset and split into root, client, and test sets
2. Create server and clients (some malicious)
3. Train the VAE model on root dataset gradients
4. Train the dual attention model for malicious client detection
5. Run federated learning with the dual attention mechanism
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the federated learning package
from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds
from federated_learning.training.training_utils import train_dual_attention

# Define missing constants
DUAL_ATTENTION_EPOCHS = 5
DUAL_ATTENTION_LEARNING_RATE = 0.001
DUAL_ATTENTION_BATCH_SIZE = 32  # Added batch size for dual attention training

# Ensure batch size is at least 2 to avoid BatchNorm issues
MIN_BATCH_SIZE = 2
if BATCH_SIZE < MIN_BATCH_SIZE:
    print(f"Warning: Increasing batch size from {BATCH_SIZE} to {MIN_BATCH_SIZE} to avoid BatchNorm issues")
    BATCH_SIZE = MIN_BATCH_SIZE

def main():
    """Main function to run the federated learning system."""
    # Set random seeds for reproducibility
    if RANDOM_SEED is not None:
        set_random_seeds(RANDOM_SEED)
        print(f"Random seeds set to {RANDOM_SEED}")
    
    # Create server and datasets
    print("\n--- Creating server and clients ---")
    server = Server()
    
    # Step 1: Load dataset
    # Determine the data path based on dataset type
    if DATASET == 'MNIST':
        data_path = MNIST_DATA_ROOT
    elif DATASET == 'ALZHEIMER':
        data_path = ALZHEIMER_DATA_ROOT
    elif DATASET == 'CIFAR10':
        data_path = CIFAR_DATA_ROOT
    else:
        data_path = './data'  # Default path
        
    # Load the dataset using the appropriate function
    root_dataset, test_dataset = load_dataset()
    
    # Create a root dataloader
    root_loader = torch.utils.data.DataLoader(
        root_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Set datasets on server
    server.set_datasets(root_loader, test_dataset)
    print("Datasets set for server")
    
    # Step 2: Pretrain global model
    print("\n--- Pretraining global model ---")
    server._pretrain_global_model()
    
    # Step 3: Create client datasets
    _, client_datasets = create_client_datasets(
        train_dataset=root_dataset,
        num_clients=NUM_CLIENTS,
        iid=not ENABLE_NON_IID,
        alpha=DIRICHLET_ALPHA if ENABLE_NON_IID else None
    )
    
    # Create malicious client flags based on ratio
    malicious_client_count = int(NUM_CLIENTS * FRACTION_MALICIOUS)
    malicious_indices = random.sample(range(NUM_CLIENTS), malicious_client_count)
    
    print(f"\nCreating {NUM_CLIENTS} clients, {malicious_client_count} will be malicious")
    print(f"Malicious client indices: {malicious_indices}")
    
    # Step 4: Create clients
    clients = []
    for i, dataset in enumerate(client_datasets):
        # Check if this client should be malicious
        is_malicious = (i in malicious_indices)
        
        # Create client
        client = Client(
            client_id=i,
            dataset=dataset,
            is_malicious=is_malicious
        )
        
        # Set attack type for malicious clients and verify it's set correctly
        if is_malicious:
            attack_type = ATTACK_TYPE
            print(f"Setting client {i} as malicious with attack type: {attack_type}")
            client.set_attack_parameters(
                attack_type=attack_type,
                scaling_factor=SCALING_FACTOR,
                partial_percent=PARTIAL_SCALING_PERCENT
            )
            print(f"Verified client {i} - is_malicious flag: {client.is_malicious}")
            print(f"Verified client {i} - has attack: {hasattr(client, 'attack')}")
            if hasattr(client, 'attack'):
                print(f"  Attack type: {client.attack.attack_type}")
                print(f"  Scaling factor: {SCALING_FACTOR}")
                print(f"  Affected percent: {PARTIAL_SCALING_PERCENT * 100:.1f}%")
        
        clients.append(client)
    
    # Set clients on server
    server.add_clients(clients)
    
    # Print malicious client indices for debugging
    print("\nClient configuration:")
    for i, client in enumerate(clients):
        status = "MALICIOUS" if client.is_malicious else "HONEST"
        print(f"Client {i}: {status}")

    # Print the server's client mappings
    print("\nServer's client mappings:")
    for i, client in enumerate(server.clients):
        status = "MALICIOUS" if client.is_malicious else "HONEST"
        print(f"Server client {i}: Client ID = {client.client_id}, Status = {status}")
    
    # Step 5: Collect root gradients (needed for VAE training)
    print("\n--- Collecting root gradients ---")
    root_gradients = server._collect_root_gradients()
    
    # Step 6: Train VAE on root gradients
    print("\n--- Training VAE on root gradients ---")
    server.vae = server.train_vae(root_gradients, vae_epochs=VAE_EPOCHS)
    
    # Step 7: Train dual attention model for malicious client detection
    print("\n--- Training dual attention model with diverse attack types ---")
    
    # Generate or load training data for dual attention
    print("Generating training data for dual attention model...")
    
    # First, extract features from honest root gradients
    honest_features = []
    for grad in root_gradients:
        # Extract features from each gradient
        features = server._compute_gradient_features(grad)
        honest_features.append(features)
    
    # Convert to tensor if not already
    if not isinstance(honest_features, torch.Tensor):
        honest_features = torch.stack(honest_features)
    
    print(f"Generated {len(honest_features)} honest feature vectors from root gradients")
    
    # Generate malicious gradients by applying ALL attack types to copies of root gradients
    print("Generating malicious gradients using multiple attack types...")
    
    # List of all attack types to simulate
    all_attack_types = [
        'scaling_attack', 
        'partial_scaling_attack', 
        'sign_flipping_attack', 
        'noise_attack', 
        'min_max_attack', 
        'min_sum_attack',
        'targeted_attack'
    ]
    
    # Parameters for different attacks
    attack_params = {
        'scaling_attack': {'scaling_factor': 15.0},
        'partial_scaling_attack': {'scaling_factor': 10.0, 'percent': 0.3},
        'sign_flipping_attack': {'percent': 0.7},
        'noise_attack': {'noise_factor': 3.0},
        'min_max_attack': {'target_class': 0},
        'min_sum_attack': {'target_weight': 0.9},
        'targeted_attack': {'target_layer': -1, 'scaling_factor': 10.0}
    }
    
    malicious_features = []
    malicious_gradients = []
    
    # Apply each attack type to each root gradient
    for attack_type in all_attack_types:
        print(f"Applying {attack_type}...")
        
        # Process a subset of root gradients (to avoid too many samples)
        gradient_subset = random.sample(root_gradients, min(10, len(root_gradients)))
        
        for grad in gradient_subset:
            # Create a copy of the gradient
            grad_copy = grad.clone().to(server.device)
            
            # Apply the attack to the gradient copy
            if attack_type == 'scaling_attack':
                scaling_factor = attack_params['scaling_attack']['scaling_factor']
                attacked_grad = grad_copy * scaling_factor
            elif attack_type == 'partial_scaling_attack':
                # Scale a percentage of the gradient
                percent = attack_params['partial_scaling_attack']['percent']
                scaling_factor = attack_params['partial_scaling_attack']['scaling_factor']
                num_elements = int(grad_copy.numel() * percent)
                indices = torch.randperm(grad_copy.numel())[:num_elements]
                flat_grad = grad_copy.view(-1)
                flat_grad[indices] *= scaling_factor
                attacked_grad = flat_grad.view_as(grad_copy)
            elif attack_type == 'sign_flipping_attack':
                # Flip signs of gradient elements
                percent = attack_params['sign_flipping_attack']['percent']
                num_elements = int(grad_copy.numel() * percent)
                indices = torch.randperm(grad_copy.numel())[:num_elements]
                flat_grad = grad_copy.view(-1)
                flat_grad[indices] *= -1
                attacked_grad = flat_grad.view_as(grad_copy)
            elif attack_type == 'noise_attack':
                # Add random noise
                noise = torch.randn_like(grad_copy) * attack_params['noise_attack']['noise_factor'] * torch.norm(grad_copy)
                attacked_grad = grad_copy + noise
            else:
                # For other attack types, apply a simple scaling as fallback
                attacked_grad = grad_copy * 5.0
            
            # Extract features from the attacked gradient
            mal_features = server._compute_gradient_features(attacked_grad, root_gradient=grad, skip_client_sim=True)
            
            # Store the features and gradient
            malicious_features.append(mal_features)
            malicious_gradients.append(attacked_grad)
    
    # Convert to tensor
    if not isinstance(malicious_features, torch.Tensor):
        malicious_features = torch.stack(malicious_features)
    
    print(f"Generated {len(malicious_features)} malicious feature vectors across {len(all_attack_types)} attack types")
    print(f"Feature dimensions - Honest: {honest_features.shape}, Malicious: {malicious_features.shape}")
    
    # Print statistical comparison between honest and malicious features
    print("\nFeature statistics comparison:")
    for i in range(honest_features.shape[1]):
        h_mean = honest_features[:, i].mean().item()
        h_std = honest_features[:, i].std().item()
        m_mean = malicious_features[:, i].mean().item()
        m_std = malicious_features[:, i].std().item()
        
        feature_names = [
            "VAE Reconstruction Error", 
            "Root Similarity", 
            "Client Similarity", 
            "Gradient Norm", 
            "Sign Consistency", 
            "Shapley Value"
        ]
        
        print(f"Feature {i+1} ({feature_names[i]}): ")
        print(f"  Honest - Mean: {h_mean:.4f}, Std: {h_std:.4f}")
        print(f"  Malicious - Mean: {m_mean:.4f}, Std: {m_std:.4f}")
        print(f"  Difference: {m_mean - h_mean:.4f} ({(m_mean - h_mean)/h_mean*100:.1f}%)")
    
    # Train the dual attention model using the training utility function
    server.dual_attention = train_dual_attention(
        honest_features=honest_features,
        malicious_features=malicious_features,
        epochs=DUAL_ATTENTION_EPOCHS,
        batch_size=DUAL_ATTENTION_BATCH_SIZE,
        lr=DUAL_ATTENTION_LEARNING_RATE,
        weight_decay=1e-4,
        device=server.device,
        verbose=True,
        early_stopping=5
    )
    
    print("Dual attention model training completed")
    print(f"Trained model has {server.dual_attention.feature_dim} input features and {server.dual_attention.hidden_dim} hidden dimensions")
    
    # Save the trained model for future use
    model_dir = "model_weights"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(server.dual_attention.state_dict(), os.path.join(model_dir, 'dual_attention.pth'))
    print(f"Saved trained dual attention model to {os.path.join(model_dir, 'dual_attention.pth')}")
    
    # Step 8: Start federated learning
    print("\n--- Starting federated learning ---")
    try:
        test_errors, round_metrics = server.train(num_rounds=GLOBAL_EPOCHS)
        
        # Step 9: Save and plot results
        print("\n--- Saving and plotting results ---")
        
        # Save round metrics
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/round_metrics_{timestamp}.json", "w") as f:
            json.dump(round_metrics, f, indent=2)
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(test_errors)), test_errors, marker='o')
        plt.title("Federated Learning Training Progress")
        plt.xlabel("Communication Round")
        plt.ylabel("Test Error")
        plt.grid(True)
        plt.savefig(f"training_progress_{timestamp}.png")
        plt.close()
        
        # Plot trust scores and weights (if available)
        if hasattr(server, 'trust_scores') and server.trust_scores is not None:
            plt.figure(figsize=(10, 6))
            trust_scores = server.trust_scores.cpu().numpy()
            client_indices = list(range(len(trust_scores)))
            
            # Mark malicious clients with different colors
            honest_indices = [i for i, client in enumerate(server.clients) if not client.is_malicious]
            malicious_indices = [i for i, client in enumerate(server.clients) if client.is_malicious]
            
            # Plot honest clients
            if honest_indices:
                plt.bar([str(i) for i in honest_indices], 
                       trust_scores[honest_indices], 
                       color='blue', 
                       label='Honest Clients')
            
            # Plot malicious clients if any exist
            if malicious_indices:
                plt.bar([str(i) for i in malicious_indices], 
                       trust_scores[malicious_indices], 
                       color='red', 
                       label='Malicious Clients')
            
            plt.xlabel('Client ID')
            plt.ylabel('Trust Score')
            plt.title('Trust Scores by Client')
            plt.legend()
            plt.savefig('trust_scores.png')
            
            # Plot weights
            if hasattr(server, 'aggregation_weights') and server.aggregation_weights is not None:
                plt.figure(figsize=(10, 6))
                weights = server.aggregation_weights.cpu().numpy()
                
                # Plot honest clients
                if honest_indices:
                    plt.bar([str(i) for i in honest_indices], 
                           weights[honest_indices], 
                           color='blue', 
                           label='Honest Clients')
                
                # Plot malicious clients if any exist
                if malicious_indices:
                    plt.bar([str(i) for i in malicious_indices], 
                           weights[malicious_indices], 
                           color='red', 
                           label='Malicious Clients')
                
                plt.xlabel('Client ID')
                plt.ylabel('Aggregation Weight')
                plt.title('Aggregation Weights by Client')
                plt.legend()
                plt.savefig('aggregation_weights.png')
        
        print("\n--- Federated learning completed successfully ---")
        
        # Print summary statistics to verify malicious client detection
        print("\n====== DETECTION SUMMARY ======")
        detected_malicious = []
        
        # Get final round metrics - fix for KeyError
        final_round = max([int(k) for k in round_metrics.keys() if isinstance(k, str)] + [k for k in round_metrics.keys() if isinstance(k, int)])
        # Access final_metrics with integer key
        final_metrics = round_metrics[final_round]
        
        # Check which clients were detected as malicious in the final round
        for client_idx, client in enumerate(server.clients):
            is_actually_malicious = client.is_malicious
            
            # Check if client was detected as malicious (significantly lower weight)
            weight = final_metrics['weights'].get(client_idx, None)
            trust_score = final_metrics['trust_scores'].get(client_idx, None)
            
            # If client participated in the final round
            if weight is not None:
                avg_weight = 1.0 / len(server.clients)
                is_detected_malicious = weight < avg_weight * 0.7  # Threshold for detection
                
                status = "✓ CORRECTLY " if is_detected_malicious == is_actually_malicious else "✗ INCORRECTLY "
                actual = "MALICIOUS" if is_actually_malicious else "HONEST"
                detected = "MALICIOUS" if is_detected_malicious else "HONEST"
                
                print(f"Client {client_idx}: {status} identified as {detected} (Actually {actual})")
                print(f"  Weight: {weight:.4f}, Trust score: {trust_score:.4f}")
                
                if is_detected_malicious:
                    detected_malicious.append(client_idx)
        
        # Calculate detection accuracy
        true_malicious = [i for i, client in enumerate(server.clients) if client.is_malicious]
        true_benign = [i for i, client in enumerate(server.clients) if not client.is_malicious]
        
        true_positives = len([i for i in detected_malicious if i in true_malicious])
        false_positives = len([i for i in detected_malicious if i not in true_malicious])
        
        if len(true_malicious) > 0:
            recall = true_positives / len(true_malicious)
        else:
            recall = 1.0  # No malicious clients to detect
            
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        
        print(f"\nDetection Results:")
        print(f"  True positives: {true_positives}/{len(true_malicious)} malicious clients detected")
        print(f"  False positives: {false_positives}/{len(true_benign)} honest clients misclassified")
        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}")
        
    except Exception as e:
        print(f"\n!!! Error during federated learning: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 