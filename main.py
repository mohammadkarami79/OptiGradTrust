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
    """Main function to run the federated learning experiment."""
    
    # Initialize results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'DATASET': DATASET,
            'MODEL': MODEL,
            'AGGREGATION_METHOD': AGGREGATION_METHOD,
            'ATTACK_TYPE': ATTACK_TYPE,
            'NUM_CLIENTS': NUM_CLIENTS,
            'FRACTION_MALICIOUS': FRACTION_MALICIOUS,
            'GLOBAL_EPOCHS': GLOBAL_EPOCHS,
            'SCALING_FACTOR': SCALING_FACTOR
        },
        'status': 'started'
    }
    
    # Step 1: Set random seed for reproducibility
    print("Setting random seed for reproducibility...")
    if RANDOM_SEED is not None:
        set_random_seeds(RANDOM_SEED)
        print(f"Random seed set to: {RANDOM_SEED}")
    
    # Step 2: Load and preprocess data
    print("\n--- Loading and preprocessing data ---")
    root_dataset, test_dataset = load_dataset()
    print(f"Root dataset size: {len(root_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Step 3: Create data loaders
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Step 4: Create server and set datasets
    print("\n--- Creating server and setting up model ---")
    server = Server()
    server.set_datasets(root_loader, test_dataset)
    
    # Pretrain global model
    server._pretrain_global_model()
    
    # Get initial accuracy
    initial_accuracy = server.evaluate_model()
    results['initial_accuracy'] = initial_accuracy
    print(f"Initial model accuracy: {initial_accuracy:.4f}")
    
    # Step 5: Create client datasets
    print("\n--- Creating client datasets ---")
    root_client_dataset, client_datasets = create_client_datasets(
        train_dataset=root_dataset,
        num_clients=NUM_CLIENTS,
        iid=not ENABLE_NON_IID,
        alpha=DIRICHLET_ALPHA if ENABLE_NON_IID else None
    )
    
    print(f"Created {len(client_datasets)} client datasets")
    for i, dataset in enumerate(client_datasets):
        print(f"Client {i}: {len(dataset)} samples")
    
    # Step 6: Create clients
    print("\n--- Creating clients ---")
    clients = []
    num_malicious = int(NUM_CLIENTS * FRACTION_MALICIOUS)
    malicious_indices = np.random.choice(NUM_CLIENTS, num_malicious, replace=False)
    
    print(f"Creating {num_malicious} malicious clients out of {NUM_CLIENTS} total clients")
    print(f"Malicious client indices: {malicious_indices}")
    
    for i in range(NUM_CLIENTS):
        is_malicious = i in malicious_indices
        
        client = Client(
            client_id=i,
            dataset=client_datasets[i], 
            is_malicious=is_malicious
        )
        
        if is_malicious:
            client.set_attack_parameters(
                attack_type=ATTACK_TYPE,
                scaling_factor=SCALING_FACTOR,
                partial_percent=PARTIAL_SCALING_PERCENT
            )
            print(f"Client {i}: MALICIOUS ({ATTACK_TYPE})")
        else:
            print(f"Client {i}: HONEST")
        
        clients.append(client)
    
    server.add_clients(clients)
    
    # Step 7: Train VAE and dual attention
    print("\n--- Training VAE on root gradients ---")
    
    # Collect root gradients for VAE training
    root_gradients = server._collect_root_gradients()
    print(f"Collected {len(root_gradients)} root gradients")
    
    # Train VAE on root gradients
    server.vae = server.train_vae(root_gradients, vae_epochs=VAE_EPOCHS)
    
    # Train dual attention model with comprehensive attack simulation
    print("\n--- Training dual attention model ---")
    
    # Extract features from honest root gradients  
    honest_features = []
    for grad in root_gradients[:min(20, len(root_gradients))]:  # Use up to 20 gradients
        features = server._compute_gradient_features(grad, skip_client_sim=True)
        honest_features.append(features)
    
    if not isinstance(honest_features, torch.Tensor):
        honest_features = torch.stack(honest_features)
    
    print(f"Extracted {len(honest_features)} honest feature vectors")
    
    # Generate malicious training data by applying ALL attack types
    print("Generating malicious training data using comprehensive attack simulation...")
    
    # Define all available attack types
    all_attack_types = [
        'scaling_attack',
        'partial_scaling_attack', 
        'sign_flipping_attack',
        'noise_attack',
        'min_max_attack',
        'min_sum_attack',
        'targeted_attack'
    ]
    
    malicious_features = []
    malicious_gradients = []
    
    # Apply each attack type to multiple root gradients
    for attack_type in all_attack_types:
        print(f"Applying {attack_type}...")
        
        for i, grad in enumerate(root_gradients[:5]):  # Use first 5 gradients per attack
            grad_copy = grad.clone()
            
            # Apply the specific attack based on type
            if attack_type == 'scaling_attack':
                attacked_grad = grad_copy * SCALING_FACTOR
                
            elif attack_type == 'partial_scaling_attack':
                # Apply scaling to only a subset of parameters
                mask = torch.rand_like(grad_copy) < PARTIAL_SCALING_PERCENT
                attacked_grad = grad_copy.clone()
                attacked_grad[mask] *= SCALING_FACTOR
                
            elif attack_type == 'sign_flipping_attack':
                attacked_grad = grad_copy * -1
                
            elif attack_type == 'noise_attack':
                noise = torch.randn_like(grad_copy) * 0.1
                attacked_grad = grad_copy + noise
                
            elif attack_type == 'min_max_attack':
                # Set gradients to extreme values
                attacked_grad = torch.where(grad_copy > 0, 
                                          torch.ones_like(grad_copy) * 10.0,
                                          torch.ones_like(grad_copy) * -10.0)
                
            elif attack_type == 'min_sum_attack':
                # Minimize the sum of gradients
                attacked_grad = grad_copy * 0.01
                
            elif attack_type == 'targeted_attack':
                # Set all gradients to a target value that would benefit the attacker
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
        
        # Get final accuracy
        final_accuracy = server.evaluate_model()
        results['final_accuracy'] = final_accuracy
        results['test_errors'] = test_errors
        results['round_metrics'] = round_metrics
        
        # Step 9: Calculate detection metrics
        print("\n--- Calculating detection metrics ---")
        
        # Get final round metrics
        final_round = max([int(k) for k in round_metrics.keys() if isinstance(k, str)] + [k for k in round_metrics.keys() if isinstance(k, int)])
        final_metrics = round_metrics[final_round]
        
        # Define client lists for consistent access
        true_malicious = [i for i, client in enumerate(server.clients) if client.is_malicious]
        true_honest = [i for i, client in enumerate(server.clients) if not client.is_malicious]
        
        # Use the actual detection results from the server
        detection_results = final_metrics.get('detection_results', {})
        
        if detection_results:
            # Use the computed detection results from the server
            true_positives = detection_results['true_positives']
            false_positives = detection_results['false_positives']
            false_negatives = detection_results['false_negatives']
            true_negatives = detection_results['true_negatives']
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (true_positives + true_negatives) / len(server.clients)
            
        else:
            # Fallback to weight-based detection (less accurate)
            detected_malicious = []
            
            # Calculate average weights and trust scores
            avg_weight = 1.0 / len(server.clients)
            detection_threshold = avg_weight * 0.7
            
            for client_idx, client in enumerate(server.clients):
                weight = final_metrics['weights'].get(client_idx, avg_weight)
                
                if weight < detection_threshold:
                    detected_malicious.append(client_idx)
            
            # Calculate metrics
            true_positives = len([i for i in detected_malicious if i in true_malicious])
            false_positives = len([i for i in detected_malicious if i not in true_malicious])
            false_negatives = len([i for i in true_malicious if i not in detected_malicious])
            true_negatives = len([i for i in true_honest if i not in detected_malicious])
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (true_positives + true_negatives) / len(server.clients)
        
        # Calculate trust score averages
        total_trust_honest = 0
        total_trust_malicious = 0
        count_honest = 0
        count_malicious = 0
        
        for client_idx, client in enumerate(server.clients):
            trust_score = final_metrics['trust_scores'].get(client_idx, 0.5)
            
            if client.is_malicious:
                total_trust_malicious += trust_score
                count_malicious += 1
            else:
                total_trust_honest += trust_score
                count_honest += 1
        
        avg_trust_honest = total_trust_honest / count_honest if count_honest > 0 else 0.0
        avg_trust_malicious = total_trust_malicious / count_malicious if count_malicious > 0 else 0.0
        
        detection_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'malicious_detection_rate': recall,
            'false_positive_rate': false_positives / len(true_honest) if len(true_honest) > 0 else 0.0,
            'avg_trust_honest': avg_trust_honest,
            'avg_trust_malicious': avg_trust_malicious
        }
        
        results['detection_metrics'] = detection_metrics
        
        # Step 10: Save results (moved to experiment runner)
        
        print("\n--- Federated learning completed successfully ---")
        results['status'] = 'completed'
        
        # Print summary
        print("\n====== EXPERIMENT SUMMARY ======")
        print(f"Initial Accuracy: {initial_accuracy:.4f}")
        print(f"Final Accuracy: {final_accuracy:.4f}")
        print(f"Improvement: {final_accuracy - initial_accuracy:.4f}")
        print(f"Detection Precision: {precision:.4f}")
        print(f"Detection Recall: {recall:.4f}")
        print(f"Detection F1-Score: {f1_score:.4f}")
        print(f"Trust Score - Honest Avg: {avg_trust_honest:.4f}")
        print(f"Trust Score - Malicious Avg: {avg_trust_malicious:.4f}")
        
    except Exception as e:
        print(f"\n!!! Error during federated learning: {str(e)}")
        results['status'] = 'failed'
        results['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return results

if __name__ == "__main__":
    main() 