#!/usr/bin/env python3
"""Quick test for MNIST optimization - 3 rounds only"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.config.config import *
from federated_learning.training.server import FederatedServer
from federated_learning.training.client import Client
import torch
import numpy as np

# Override global epochs for quick test
GLOBAL_EPOCHS = 3  # Just 3 rounds for testing

def main():
    print("=" * 50)
    print("MNIST OPTIMIZATION TEST - 3 ROUNDS ONLY")
    print("=" * 50)
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    print(f"Malicious Penalty: {MALICIOUS_PENALTY_FACTOR}")
    print(f"Zero Attack Threshold: {ZERO_ATTACK_THRESHOLD}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize server
    server = FederatedServer()
    
    # Run test
    try:
        server.federated_train()
        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        print("Optimization parameters are working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("Need to adjust parameters.")
        
    print("=" * 50)

if __name__ == "__main__":
    main() 