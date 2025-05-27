#!/usr/bin/env python3
"""
Test to ensure MNIST + CNN still works correctly after ALZHEIMER fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure before imports
import federated_learning.config.config as config
config.DATASET = 'MNIST'
config.MODEL = 'CNN'
config.NUM_CLIENTS = 3
config.FRACTION_MALICIOUS = 0.33
config.GLOBAL_EPOCHS = 2
config.LOCAL_EPOCHS_CLIENT = 1
config.ATTACK_TYPE = 'scaling_attack'
config.AGGREGATION_METHOD = 'fedavg'
config.RL_AGGREGATION_METHOD = 'hybrid'
config.BATCH_SIZE = 32  # Increased batch size to avoid BatchNorm issues
config.FAST_MODE = True

import torch
import torch.nn as nn
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.utils.data_utils import get_dataset, create_federated_dataset

def test_mnist_cnn():
    """Test MNIST dataset with CNN model."""
    print("MNIST + CNN COMPATIBILITY TEST")
    print("=" * 50)
    
    print(f"Dataset: {config.DATASET}")
    print(f"Model: {config.MODEL}")
    print(f"RL Aggregation: {config.RL_AGGREGATION_METHOD}")
    print(f"Attack Type: {config.ATTACK_TYPE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    
    # Test 1: Create server
    print("\nTest 1: Creating server...")
    try:
        server = Server()
        print("✅ Server created successfully")
        print(f"Global model type: {type(server.global_model)}")
        
        # Check model architecture with proper batch size
        sample_input = torch.randn(4, 1, 28, 28).to(server.device)  # Use batch size 4
        with torch.no_grad():
            server.global_model.eval()  # Set to eval mode to avoid BatchNorm training issues
            output = server.global_model(sample_input)
            print(f"Model output shape: {output.shape}")
            print(f"Expected output shape: (4, 10)")
            
        if output.shape[1] == 10:
            print("✅ Model architecture correct")
        else:
            print(f"❌ Model architecture incorrect. Expected 10 classes, got {output.shape[1]}")
            return False
            
    except Exception as e:
        print(f"❌ Server creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Load dataset
    print("\nTest 2: Loading MNIST dataset...")
    try:
        train_dataset, test_dataset = get_dataset('MNIST')
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
        
        # Check a sample
        sample_data, sample_label = train_dataset[0]
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample label: {sample_label}")
        print(f"Expected data shape: (1, 28, 28)")
        
        if sample_data.shape == (1, 28, 28):
            print("✅ Data format correct")
        else:
            print(f"❌ Data format incorrect. Expected (1, 28, 28), got {sample_data.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {str(e)}")
        return False
    
    # Test 3: Create federated dataset
    print("\nTest 3: Creating federated dataset...")
    try:
        federated_train_dataset, root_dataset, test_loader = create_federated_dataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_clients=3,
            root_dataset_ratio=0.1,
            iid=False,
            dirichlet_alpha=0.5
        )
        print(f"✅ Federated dataset created")
        print(f"Root dataset size: {len(root_dataset)}")
        print(f"Client datasets sizes: {[len(dataset) for dataset in federated_train_dataset]}")
        
    except Exception as e:
        print(f"❌ Federated dataset creation failed: {str(e)}")
        return False
    
    # Test 4: Create clients
    print("\nTest 4: Creating clients...")
    try:
        clients = []
        for i in range(3):
            is_malicious = i == 2  # Last client is malicious
            client = Client(
                client_id=i,
                dataset=federated_train_dataset[i],
                is_malicious=is_malicious,
                local_epochs=1
            )
            clients.append(client)
            
            print(f"Client {i}: Created ({'malicious' if is_malicious else 'honest'})")
            print(f"  Model type: {type(client.model)}")
            print(f"  Dataset size: {len(federated_train_dataset[i])}")
            
        print("✅ All clients created successfully")
        
    except Exception as e:
        print(f"❌ Client creation failed: {str(e)}")
        return False
    
    # Test 5: Test model forward pass with MNIST data
    print("\nTest 5: Testing model forward pass...")
    try:
        # Get a batch from the first client
        client = clients[0]
        data_loader = client.train_loader
        
        for batch_data, batch_labels in data_loader:
            print(f"Batch data shape: {batch_data.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
            
            # Skip batches that are too small for BatchNorm
            if batch_data.shape[0] < 2:
                print("Skipping small batch for BatchNorm compatibility")
                continue
            
            # Move to device
            batch_data = batch_data.to(server.device)
            batch_labels = batch_labels.to(server.device)
            
            # Test forward pass
            with torch.no_grad():
                server.global_model.eval()
                output = server.global_model(batch_data)
                print(f"Model output shape: {output.shape}")
                print(f"Expected: ({batch_data.shape[0]}, 10)")
                
                if output.shape == (batch_data.shape[0], 10):
                    print("✅ Forward pass successful")
                else:
                    print(f"❌ Forward pass failed. Wrong output shape")
                    return False
            break  # Only test first valid batch
            
    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test client training
    print("\nTest 6: Testing client training...")
    try:
        client = clients[0]
        
        # Train for one step
        gradient, features = client.train(server.global_model, round_idx=0)
        
        if gradient is not None:
            print(f"✅ Client training successful")
            print(f"Gradient shape: {gradient.shape}")
            print(f"Gradient norm: {torch.norm(gradient).item():.4f}")
            
            if features is not None:
                print(f"Features shape: {features.shape}")
            else:
                print("Features: None")
        else:
            print("❌ Client training failed - no gradient returned")
            return False
            
    except Exception as e:
        print(f"❌ Client training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 7: Test malicious client with attack
    print("\nTest 7: Testing malicious client...")
    try:
        malicious_client = clients[2]  # Last client is malicious
        
        # Train malicious client
        gradient, features = malicious_client.train(server.global_model, round_idx=0)
        
        if gradient is not None:
            print(f"✅ Malicious client training successful")
            print(f"Gradient shape: {gradient.shape}")
            print(f"Gradient norm: {torch.norm(gradient).item():.4f}")
            
            # Check if attack was applied (should have larger gradient norm)
            if hasattr(malicious_client, 'original_gradient'):
                original_norm = torch.norm(malicious_client.original_gradient).item()
                attacked_norm = torch.norm(gradient).item()
                print(f"Original gradient norm: {original_norm:.4f}")
                print(f"Attacked gradient norm: {attacked_norm:.4f}")
                print(f"Attack amplification: {attacked_norm/original_norm:.2f}x")
                
                if attacked_norm > original_norm:
                    print("✅ Attack successfully applied")
                else:
                    print("⚠️  Attack might not have been applied effectively")
        else:
            print("❌ Malicious client training failed")
            return False
            
    except Exception as e:
        print(f"❌ Malicious client training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED - MNIST + CNN STILL WORKING!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_mnist_cnn()
    if success:
        print("\n🎉 MNIST + CNN compatibility is maintained!")
    else:
        print("\n❌ MNIST + CNN compatibility has been broken.")
        sys.exit(1) 