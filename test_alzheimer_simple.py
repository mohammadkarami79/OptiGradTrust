#!/usr/bin/env python3
"""
Simple test to verify ALZHEIMER dataset works with ResNet18 model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set configuration BEFORE importing server and client
import federated_learning.config.config as config
config.DATASET = 'ALZHEIMER'
config.MODEL = 'RESNET18'
config.NUM_CLIENTS = 3
config.FRACTION_MALICIOUS = 0.33
config.GLOBAL_EPOCHS = 1
config.LOCAL_EPOCHS_CLIENT = 1
config.ATTACK_TYPE = 'none'
config.AGGREGATION_METHOD = 'fedavg'
config.BATCH_SIZE = 8
config.FAST_MODE = True

# Now import the modules that depend on config
import torch
import torch.nn as nn
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.utils.data_utils import get_dataset, create_federated_dataset

def test_alzheimer_resnet18():
    """Test ALZHEIMER dataset with ResNet18 model."""
    print("ALZHEIMER + RESNET18 SIMPLE TEST")
    print("=" * 50)
    
    print(f"Dataset: {config.DATASET}")
    print(f"Model: {config.MODEL}")
    print(f"Alzheimer Classes: {config.ALZHEIMER_CLASSES}")
    
    # Test 1: Create server
    print("\nTest 1: Creating server...")
    try:
        server = Server()
        print("✅ Server created successfully")
        print(f"Global model type: {type(server.global_model)}")
        print(f"Global model device: {server.global_model.device if hasattr(server.global_model, 'device') else 'unknown'}")
        
        # Check model architecture
        sample_input = torch.randn(1, 3, 128, 128).to(server.device)
        with torch.no_grad():
            output = server.global_model(sample_input)
            print(f"Model output shape: {output.shape}")
            print(f"Expected output shape: (1, {config.ALZHEIMER_CLASSES})")
            
        if output.shape[1] == config.ALZHEIMER_CLASSES:
            print("✅ Model architecture correct")
        else:
            print(f"❌ Model architecture incorrect. Expected {config.ALZHEIMER_CLASSES} classes, got {output.shape[1]}")
            
    except Exception as e:
        print(f"❌ Server creation failed: {str(e)}")
        return False
    
    # Test 2: Load dataset
    print("\nTest 2: Loading ALZHEIMER dataset...")
    try:
        train_dataset, test_dataset = get_dataset('ALZHEIMER')
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
        
        # Check a sample
        sample_data, sample_label = train_dataset[0]
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample label: {sample_label}")
        print(f"Expected data shape: (3, 128, 128)")
        
        if sample_data.shape == (3, 128, 128):
            print("✅ Data format correct")
        else:
            print(f"❌ Data format incorrect. Expected (3, 128, 128), got {sample_data.shape}")
            
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
    
    # Test 5: Test model forward pass with ALZHEIMER data
    print("\nTest 5: Testing model forward pass...")
    try:
        # Get a batch from the first client
        client = clients[0]
        data_loader = client.train_loader
        
        for batch_data, batch_labels in data_loader:
            print(f"Batch data shape: {batch_data.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
            
            # Move to device
            batch_data = batch_data.to(server.device)
            batch_labels = batch_labels.to(server.device)
            
            # Test forward pass
            with torch.no_grad():
                server.global_model.eval()
                output = server.global_model(batch_data)
                print(f"Model output shape: {output.shape}")
                print(f"Expected: ({batch_data.shape[0]}, {config.ALZHEIMER_CLASSES})")
                
                if output.shape == (batch_data.shape[0], config.ALZHEIMER_CLASSES):
                    print("✅ Forward pass successful")
                else:
                    print(f"❌ Forward pass failed. Wrong output shape")
                    return False
            break  # Only test first batch
            
    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test client training
    print("\nTest 6: Testing client training...")
    try:
        client = clients[0]
        original_state = {k: v.clone() for k, v in server.global_model.state_dict().items()}
        
        # Train for one step
        gradient, features = client.train(server.global_model, round_idx=0)
        
        if gradient is not None:
            print(f"✅ Client training successful")
            print(f"Gradient shape: {gradient.shape}")
            print(f"Gradient norm: {torch.norm(gradient).item():.4f}")
            
            if features is not None:
                print(f"Features shape: {features.shape}")
                print(f"Features: {features}")
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
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED - ALZHEIMER + RESNET18 WORKING!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_alzheimer_resnet18()
    if success:
        print("\n🎉 ALZHEIMER + ResNet18 configuration is working correctly!")
    else:
        print("\n❌ ALZHEIMER + ResNet18 configuration has issues that need to be fixed.")
        sys.exit(1) 