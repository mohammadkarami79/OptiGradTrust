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
import argparse
import importlib

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the federated learning package
from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds
from federated_learning.training.training_utils import train_dual_attention
from federated_learning.utils.data_utils import create_federated_dataset, get_dataset
from federated_learning.utils.attack_utils import apply_attack
from federated_learning.models.vae import VAE, GradientVAE
from federated_learning.models.attention import DualAttention

# Define missing constants
DUAL_ATTENTION_EPOCHS = 5
DUAL_ATTENTION_LEARNING_RATE = 0.001
DUAL_ATTENTION_BATCH_SIZE = 32  # Added batch size for dual attention training

# Ensure batch size is at least 2 to avoid BatchNorm issues
MIN_BATCH_SIZE = 2
if BATCH_SIZE < MIN_BATCH_SIZE:
    print(f"Warning: Increasing batch size from {BATCH_SIZE} to {MIN_BATCH_SIZE} to avoid BatchNorm issues")
    BATCH_SIZE = MIN_BATCH_SIZE

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning with Dual Attention')
    
    # Basic parameters
    parser.add_argument('--dataset', type=str, default=DATASET, 
                        choices=['MNIST', 'CIFAR10', 'ALZHEIMER'],
                        help='Dataset to use for training')
    parser.add_argument('--model', type=str, default=MODEL, 
                        choices=['CNN', 'RESNET18', 'RESNET50'],
                        help='Model architecture')
    parser.add_argument('--num_clients', type=int, default=NUM_CLIENTS,
                        help='Number of clients')
    parser.add_argument('--malicious_ratio', type=float, default=FRACTION_MALICIOUS,
                        help='Ratio of malicious clients')
    parser.add_argument('--global_epochs', type=int, default=GLOBAL_EPOCHS,
                        help='Number of global training rounds')
    parser.add_argument('--local_epochs', type=int, default=LOCAL_EPOCHS_CLIENT,
                        help='Number of local training epochs per client')
    parser.add_argument('--attack_type', type=str, default=ATTACK_TYPE,
                        choices=['scaling_attack', 'partial_scaling_attack', 'sign_flipping_attack',
                                'noise_attack', 'targeted_parameters', 'label_flipping',
                                'min_max', 'min_sum', 'none'],
                        help='Type of attack for malicious clients')
    
    # Aggregation method
    parser.add_argument('--aggregation', type=str, default=AGGREGATION_METHOD,
                        choices=['fedavg', 'fedavg_with_trust', 'fedprox', 'fedbn', 'fedadmm'],
                        help='Aggregation method')
    
    # Detection parameters
    parser.add_argument('--enable_dual_attention', action='store_true', default=ENABLE_DUAL_ATTENTION,
                        help='Enable dual attention for trust score computation')
    parser.add_argument('--enable_vae', action='store_true', default=ENABLE_VAE,
                        help='Enable VAE for gradient anomaly detection')
    parser.add_argument('--enable_shapley', action='store_true', default=ENABLE_SHAPLEY,
                        help='Enable Shapley value computation')
    
    # Data distribution
    parser.add_argument('--non_iid', action='store_true', default=ENABLE_NON_IID,
                        help='Use non-IID data distribution')
    parser.add_argument('--dirichlet_alpha', type=float, default=DIRICHLET_ALPHA,
                        help='Dirichlet concentration parameter (lower = more non-IID)')
    
    # RL-based aggregation
    parser.add_argument('--rl_aggregation', type=str, default=RL_AGGREGATION_METHOD,
                        choices=['dual_attention', 'rl_actor_critic', 'hybrid'],
                        help='RL-based aggregation method')
    parser.add_argument('--warmup_rounds', type=int, default=RL_WARMUP_ROUNDS,
                        help='Number of rounds to use dual attention before starting RL')
    parser.add_argument('--ramp_up_rounds', type=int, default=RL_RAMP_UP_ROUNDS,
                        help='Number of rounds to blend dual attention with RL')
    
    # Performance and misc
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LR,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_cuda', action='store_true', default=torch.cuda.is_available(),
                        help='Use CUDA if available')
    
    # Fast mode for testing (skips time-consuming operations)
    parser.add_argument('--fast_mode', action='store_true', default=False,
                        help='Skip time-consuming operations for faster testing')
    
    args = parser.parse_args()
    return args

def update_config(args):
    """Update the global configuration with command-line arguments."""
    # Update basic config
    global DATASET, MODEL, NUM_CLIENTS, FRACTION_MALICIOUS, GLOBAL_EPOCHS
    global LOCAL_EPOCHS_CLIENT, LOCAL_EPOCHS_ROOT, ATTACK_TYPE, AGGREGATION_METHOD, BATCH_SIZE, LR, SEED
    
    DATASET = args.dataset
    MODEL = args.model
    NUM_CLIENTS = args.num_clients
    FRACTION_MALICIOUS = args.malicious_ratio
    GLOBAL_EPOCHS = args.global_epochs
    LOCAL_EPOCHS_CLIENT = args.local_epochs
    ATTACK_TYPE = args.attack_type
    AGGREGATION_METHOD = args.aggregation
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    SEED = args.seed
    
    # Update detection config
    global ENABLE_DUAL_ATTENTION, ENABLE_VAE, ENABLE_SHAPLEY
    ENABLE_DUAL_ATTENTION = args.enable_dual_attention
    ENABLE_VAE = args.enable_vae
    ENABLE_SHAPLEY = args.enable_shapley
    
    # Update data distribution config
    global ENABLE_NON_IID, DIRICHLET_ALPHA
    ENABLE_NON_IID = args.non_iid
    DIRICHLET_ALPHA = args.dirichlet_alpha
    
    # Update RL-based aggregation config
    global RL_AGGREGATION_METHOD, RL_WARMUP_ROUNDS, RL_RAMP_UP_ROUNDS
    RL_AGGREGATION_METHOD = args.rl_aggregation
    RL_WARMUP_ROUNDS = args.warmup_rounds
    RL_RAMP_UP_ROUNDS = args.ramp_up_rounds
    
    # Recalculate derived parameters
    global NUM_MALICIOUS
    NUM_MALICIOUS = int(NUM_CLIENTS * FRACTION_MALICIOUS)
    
    # Set device based on CUDA availability
    global device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    # Update device in all necessary places
    global DEVICE
    DEVICE = device
    
    # Fast mode configuration
    global FAST_MODE
    FAST_MODE = args.fast_mode
    
    if FAST_MODE:
        print("\n=== FAST MODE ENABLED ===")
        print("- Using minimal epochs for testing")
        print("- Skipping time-consuming operations if possible")
        print("- Loading pretrained models if available")
        print("=============================\n")
        
        # Reduce epochs in fast mode
        if GLOBAL_EPOCHS > 2:
            print(f"Reducing global epochs from {GLOBAL_EPOCHS} to 2 in fast mode")
            GLOBAL_EPOCHS = 2
        if LOCAL_EPOCHS_CLIENT > 2:
            print(f"Reducing local epochs from {LOCAL_EPOCHS_CLIENT} to 2 in fast mode")
            LOCAL_EPOCHS_CLIENT = 2
        if LOCAL_EPOCHS_ROOT > 1:
            print(f"Reducing root pretraining epochs from {LOCAL_EPOCHS_ROOT} to 1 in fast mode")
            LOCAL_EPOCHS_ROOT = 1
    
    # Print updated configuration
    print("\n=== Updated Configuration ===")
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL}")
    print(f"Clients: {NUM_CLIENTS}, Malicious: {NUM_MALICIOUS} ({FRACTION_MALICIOUS*100:.1f}%)")
    print(f"Global Epochs: {GLOBAL_EPOCHS}, Local Epochs: {LOCAL_EPOCHS_CLIENT}")
    print(f"Attack Type: {ATTACK_TYPE}")
    print(f"Aggregation Method: {AGGREGATION_METHOD}")
    print(f"RL Aggregation Method: {RL_AGGREGATION_METHOD}")
    if RL_AGGREGATION_METHOD == 'hybrid':
        print(f"  Warmup Rounds: {RL_WARMUP_ROUNDS}")
        print(f"  Ramp-up Rounds: {RL_RAMP_UP_ROUNDS}")
    print(f"Dual Attention: {ENABLE_DUAL_ATTENTION}, VAE: {ENABLE_VAE}, Shapley: {ENABLE_SHAPLEY}")
    print(f"Non-IID Data: {ENABLE_NON_IID}, Dirichlet Alpha: {DIRICHLET_ALPHA}")
    print(f"Device: {device}")
    print("============================\n")

    # After updating local globals, propagate to config module globals
    cfg = importlib.import_module('federated_learning.config.config')
    cfg.DATASET = DATASET
    cfg.MODEL = MODEL
    cfg.NUM_CLIENTS = NUM_CLIENTS
    cfg.FRACTION_MALICIOUS = FRACTION_MALICIOUS
    cfg.GLOBAL_EPOCHS = GLOBAL_EPOCHS
    cfg.LOCAL_EPOCHS_CLIENT = LOCAL_EPOCHS_CLIENT
    cfg.ATTACK_TYPE = ATTACK_TYPE
    cfg.AGGREGATION_METHOD = AGGREGATION_METHOD
    cfg.BATCH_SIZE = BATCH_SIZE
    cfg.LR = LR
    cfg.SEED = SEED
    cfg.ENABLE_DUAL_ATTENTION = ENABLE_DUAL_ATTENTION
    cfg.ENABLE_VAE = ENABLE_VAE
    cfg.ENABLE_SHAPLEY = ENABLE_SHAPLEY
    cfg.ENABLE_NON_IID = ENABLE_NON_IID
    cfg.DIRICHLET_ALPHA = DIRICHLET_ALPHA
    cfg.RL_AGGREGATION_METHOD = RL_AGGREGATION_METHOD
    cfg.RL_WARMUP_ROUNDS = RL_WARMUP_ROUNDS
    cfg.RL_RAMP_UP_ROUNDS = RL_RAMP_UP_ROUNDS
    cfg.FAST_MODE = FAST_MODE
    cfg.device = device

def main():
    """Main function to run the federated learning system."""
    # Parse command-line arguments
    args = parse_args()
    
    # Update configuration
    update_config(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Create server
    server = Server()
    
    # Get dataset
    train_dataset, test_dataset = get_dataset(DATASET)
    
    # Create federated dataset
    federated_train_dataset, root_dataset, test_loader = create_federated_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_clients=NUM_CLIENTS,
        root_dataset_ratio=ROOT_DATASET_RATIO,
        iid=not ENABLE_NON_IID,
        dirichlet_alpha=DIRICHLET_ALPHA
    )
    
    # Set server datasets
    root_loader = torch.utils.data.DataLoader(
        root_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS if NUM_WORKERS > 0 else 0,
        pin_memory=PIN_MEMORY
    )
    server.set_datasets(root_loader, test_dataset)
    
    # Pretrain global model on root dataset if enabled
    if LOCAL_EPOCHS_ROOT > 0:
        server._pretrain_global_model()
    
    # Collect root gradients if using VAE or dual attention
    if ENABLE_VAE or ENABLE_DUAL_ATTENTION:
        # Check if we can load saved gradients in fast mode
        saved_gradients_path = os.path.join('model_weights', f'root_gradients_{DATASET}_{MODEL}.pt')
        
        if FAST_MODE and os.path.exists(saved_gradients_path):
            print(f"Fast mode: Loading saved root gradients from {saved_gradients_path}")
            try:
                server.root_gradients = torch.load(saved_gradients_path)
                print(f"Loaded {len(server.root_gradients)} saved root gradients")
            except Exception as e:
                print(f"Error loading saved gradients: {str(e)}")
                print("Collecting new root gradients...")
                server.root_gradients = server._collect_root_gradients()
                # Save for future use
                os.makedirs(os.path.dirname(saved_gradients_path), exist_ok=True)
                torch.save(server.root_gradients, saved_gradients_path)
        else:
            print("Collecting root gradients...")
            server.root_gradients = server._collect_root_gradients()
            # Save for future use
            if len(server.root_gradients) > 0:
                os.makedirs(os.path.dirname(saved_gradients_path), exist_ok=True)
                torch.save(server.root_gradients, saved_gradients_path)
                print(f"Saved {len(server.root_gradients)} root gradients for future use")
        
        # Train VAE if enabled
        if ENABLE_VAE and server.root_gradients:
            saved_vae_path = os.path.join('model_weights', f'vae_{DATASET}_{MODEL}.pth')
            
            if FAST_MODE and os.path.exists(saved_vae_path):
                print(f"Fast mode: Loading saved VAE from {saved_vae_path}")
                try:
                    server.vae = server._create_vae()
                    server.vae.load_state_dict(torch.load(saved_vae_path))
                    server.vae = server.vae.to(server.device)
                    print("Loaded saved VAE model")
                except Exception as e:
                    print(f"Error loading saved VAE: {str(e)}")
                    print("\n=== Training VAE on Root Gradients ===")
                    server.vae = server.train_vae(server.root_gradients, vae_epochs=VAE_EPOCHS)
                    # Save for future use
                    os.makedirs(os.path.dirname(saved_vae_path), exist_ok=True)
                    torch.save(server.vae.state_dict(), saved_vae_path)
            else:
                print("\n=== Training VAE on Root Gradients ===")
                server.vae = server.train_vae(server.root_gradients, vae_epochs=VAE_EPOCHS)
                # Save for future use
                os.makedirs(os.path.dirname(saved_vae_path), exist_ok=True)
                torch.save(server.vae.state_dict(), saved_vae_path)
            
            print("VAE training completed")
    
    # Create clients
    clients = []
    # First create non-malicious clients
    for i in range(NUM_CLIENTS - NUM_MALICIOUS):
        client = Client(
            client_id=i,
            dataset=federated_train_dataset[i],
            is_malicious=False,
            local_epochs=LOCAL_EPOCHS_CLIENT
        )
        clients.append(client)
    
    # Then create malicious clients
    for i in range(NUM_CLIENTS - NUM_MALICIOUS, NUM_CLIENTS):
        client = Client(
            client_id=i,
            dataset=federated_train_dataset[i],
            is_malicious=True,
            local_epochs=LOCAL_EPOCHS_CLIENT
        )
        
        # Apply attack to malicious clients
        if ATTACK_TYPE != 'none':
            print(f"Applying {ATTACK_TYPE} attack to client {i}")
            apply_attack(client, ATTACK_TYPE)
        
        clients.append(client)
    
    # Add clients to server
    server.add_clients(clients)
    
    # Train dual attention on root gradients if enabled
    if ENABLE_DUAL_ATTENTION and server.root_gradients:
        saved_da_path = os.path.join('model_weights', f'dual_attention_{DATASET}_{MODEL}.pth')
        
        if FAST_MODE and os.path.exists(saved_da_path):
            print(f"Fast mode: Loading saved dual attention model from {saved_da_path}")
            try:
                server.dual_attention = server._create_dual_attention()
                server.dual_attention.load_state_dict(torch.load(saved_da_path))
                server.dual_attention = server.dual_attention.to(server.device)
                print("Loaded saved dual attention model")
            except Exception as e:
                print(f"Error loading saved dual attention model: {str(e)}")
                # Continue with normal training
                from federated_learning.training.training_utils import train_dual_attention
                
                # Create features from root gradients
                print("\n=== Preparing Dual Attention Training Data ===")
                root_features = []
                
                # Extract features from root gradients
                for grad in server.root_gradients:
                    features = server._compute_gradient_features(grad)
                    root_features.append(features)
                
                # Convert to tensor
                if root_features:
                    root_features = torch.stack(root_features)
                    print(f"Root features shape: {root_features.shape}")
                    
                    # Train dual attention model
                    print("\n=== Training Dual Attention Model ===")
                    server.dual_attention = train_dual_attention(
                        root_features,
                        device=server.device
                    )
                    # Save trained model
                    os.makedirs(os.path.dirname(saved_da_path), exist_ok=True)
                    torch.save(server.dual_attention.state_dict(), saved_da_path)
                    print("Dual attention training completed")
        else:
            from federated_learning.training.training_utils import train_dual_attention
            
            # Create features from root gradients
            print("\n=== Preparing Dual Attention Training Data ===")
            root_features = []
            
            # Extract features from root gradients
            for grad in server.root_gradients:
                features = server._compute_gradient_features(grad)
                root_features.append(features)
            
            # Convert to tensor
            if root_features:
                root_features = torch.stack(root_features)
                print(f"Root features shape: {root_features.shape}")
                
                # Train dual attention model
                print("\n=== Training Dual Attention Model ===")
                server.dual_attention = train_dual_attention(
                    root_features,
                    device=server.device
                )
                # Save trained model
                os.makedirs(os.path.dirname(saved_da_path), exist_ok=True)
                torch.save(server.dual_attention.state_dict(), saved_da_path)
                print("Dual attention training completed")
    
    # Pre-train RL actor-critic if using hybrid mode
    if RL_AGGREGATION_METHOD in ['rl_actor_critic', 'hybrid']:
        from federated_learning.training.rl_training import pretrain_rl_model
        
        print("\n=== Pre-training RL Actor-Critic Model ===")
        server.actor_critic = pretrain_rl_model(
            server.actor_critic,
            federated_train_dataset,
            device=server.device
        )
        print("RL pre-training completed")
    
    # Train using federated learning
    print("\n=== Starting Federated Learning Training ===")
    test_errors, round_metrics = server.train(num_rounds=GLOBAL_EPOCHS)
    
    # Save training metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = f"training_metrics_{timestamp}.pt"
    torch.save(round_metrics, metrics_file)
    print(f"Training metrics saved to {metrics_file}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(test_errors, 'b-', label='Test Error')
    plt.title('Test Error Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_file = f"training_progress_{timestamp}.png"
    plt.savefig(plot_file)
    print(f"Training progress plot saved to {plot_file}")
    
    # Print summary and statistics
    print("\n=== Training Summary ===")
    print(f"Dataset: {DATASET}, Model: {MODEL}")
    print(f"Total clients: {NUM_CLIENTS}, Malicious clients: {NUM_MALICIOUS}")
    print(f"Attack type: {ATTACK_TYPE}")
    print(f"Aggregation method: {AGGREGATION_METHOD}")
    if RL_AGGREGATION_METHOD in ['rl_actor_critic', 'hybrid']:
        print(f"RL aggregation method: {RL_AGGREGATION_METHOD}")
        if RL_AGGREGATION_METHOD == 'hybrid':
            print(f"  Warmup rounds: {RL_WARMUP_ROUNDS}")
            print(f"  Ramp-up rounds: {RL_RAMP_UP_ROUNDS}")
    print(f"Initial test error: {test_errors[0]:.4f}")
    print(f"Final test error: {test_errors[-1]:.4f}")
    print(f"Error reduction: {test_errors[0] - test_errors[-1]:.4f} ({(1 - test_errors[-1]/test_errors[0]) * 100:.2f}%)")
    print("============================")
    
    return test_errors, round_metrics

if __name__ == "__main__":
    main() 