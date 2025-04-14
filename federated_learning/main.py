import torch
import random
import numpy as np
from federated_learning.config.config import *
from federated_learning.data.dataset import load_dataset, split_dataset, create_root_dataset
from federated_learning.training.server import Server

def print_config():
    print("\n=== Configuration ===")
    print(f"Device: {device}")
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Fraction of malicious clients: {FRACTION_MALICIOUS}")
    print(f"Number of malicious clients: {NUM_MALICIOUS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Global epochs: {GLOBAL_EPOCHS}")
    print(f"Local epochs (root): {LOCAL_EPOCHS_ROOT}")
    print(f"Local epochs (client): {LOCAL_EPOCHS_CLIENT}")
    print(f"Attack type: {ATTACK_TYPE}")
    print(f"Dataset: {DATASET}")
    print(f"Data distribution: {DATA_DISTRIBUTION}")
    if DATA_DISTRIBUTION == 'label_skew':
        print(f"Q value: {Q}")
    elif DATA_DISTRIBUTION == 'dirichlet':
        print(f"Dirichlet alpha: {DIRICHLET_ALPHA}")
    print(f"Root dataset size: {ROOT_DATASET_SIZE}")
    print(f"Bias probability: {BIAS_PROBABILITY}")
    print(f"Bias class: {BIAS_CLASS}\n")

def main():
    print("=== Starting Federated Learning Training ===")
    print_config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    print("Random seeds set for reproducibility")
    
    print("\n=== Loading and Preparing Data ===")
    # Load and prepare data
    print("Downloading and loading dataset...")
    full_dataset, test_dataset, num_classes, input_channels = load_dataset()
    print(f"Dataset loaded: {len(full_dataset)} training samples, {len(test_dataset)} test samples")
    print(f"Number of classes: {num_classes}, Input channels: {input_channels}")
    
    print("\nCreating root dataset...")
    root_dataset = create_root_dataset(full_dataset, num_classes)
    print(f"Root dataset created with {len(root_dataset)} samples")
    
    print("\nSplitting remaining data...")
    remaining_indices = list(set(range(len(full_dataset))) - set(root_dataset.indices))
    remaining_dataset = torch.utils.data.Subset(full_dataset, remaining_indices)
    print(f"Remaining dataset size: {len(remaining_dataset)} samples")
    
    print("\nSplitting data among clients...")
    client_datasets = split_dataset(remaining_dataset, num_classes)
    for i, dataset in enumerate(client_datasets):
        print(f"Client {i}: {len(dataset)} samples")
    print("Data preparation completed.")
    
    print("\n=== Initializing Server and Starting Training ===")
    # Initialize and run federated learning
    server = Server(root_dataset, client_datasets, test_dataset)
    print("Server initialized, starting training...")
    server.train()

if __name__ == "__main__":
    main() 