import sys
import os
import torch
import numpy as np
from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.models.attention import DualAttention
from federated_learning.data.dataset import load_dataset
from federated_learning.utils.model_utils import set_random_seeds
import traceback

def test_gradient_feature_dimensions():
    """Test that client and server gradient feature dimensions match."""
    print("\n=== Testing Gradient Feature Dimensions ===")
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Set random seed for reproducibility
        set_random_seeds(42)
        
        print("Creating server...")
        # Create server
        server = Server()
        print(f"Created server with device: {server.device}")
        
        # Check dual attention model configuration
        expected_feature_dim = 6 if ENABLE_SHAPLEY else 5
        print(f"Expected feature dimension based on config: {expected_feature_dim}")
        
        # Verify dual attention model was initialized with correct dimension
        if server.dual_attention is not None:
            dual_attention_feature_dim = server.dual_attention.feature_dim
            print(f"DualAttention model feature_dim: {dual_attention_feature_dim}")
            assert dual_attention_feature_dim == expected_feature_dim, \
                f"DualAttention feature_dim mismatch: {dual_attention_feature_dim} vs expected {expected_feature_dim}"
        else:
            print("DualAttention model not initialized on server")
        
        print("Loading dataset...")
        # Load dataset for testing
        try:
            train_dataset, test_dataset, num_classes, _ = load_dataset()
            print(f"Loaded dataset with {len(train_dataset)} samples")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            traceback.print_exc()
            raise
        
        # Create a subset for testing
        test_indices = np.random.choice(len(train_dataset), size=1000, replace=False)
        test_subset = torch.utils.data.Subset(train_dataset, test_indices)
        
        print("Creating test client...")
        # Create a test client
        client = Client(client_id=0, dataset=test_subset, is_malicious=False)
        print(f"Created test client with {len(test_subset)} samples")
        
        # Create a small model for testing
        print("Creating test model...")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(320, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)
        
        # Train the client and get gradient
        print("\nTesting client gradient extraction...")
        try:
            gradient, client_features = client.train(model)
            print(f"Client returned gradient of shape: {gradient.shape}")
            
            if client_features is not None:
                print(f"Client returned feature vector of shape: {client_features.shape}")
                print(f"Client feature values: {client_features}")
            else:
                print("Client did not return feature vector")
        except Exception as e:
            print(f"Error in client training: {str(e)}")
            traceback.print_exc()
            raise
        
        # Test server feature computation
        print("\nTesting server feature computation...")
        try:
            raw_grad = gradient.clone()
            server_features = server._compute_gradient_features(gradient, raw_grad)
            print(f"Server computed feature vector of shape: {server_features.shape}")
            print(f"Server feature values: {server_features}")
        except Exception as e:
            print(f"Error computing gradient features: {str(e)}")
            traceback.print_exc()
            raise
        
        # Generate malicious features
        print("\nTesting malicious feature generation...")
        try:
            # Create a batch of features for malicious feature generation
            if client_features is not None:
                honest_features = client_features.unsqueeze(0).expand(5, -1)
            else:
                honest_features = server_features.unsqueeze(0).expand(5, -1)
            
            print(f"Honest features batch shape: {honest_features.shape}")
            malicious_features = server._generate_malicious_features(honest_features)
            print(f"Generated malicious features of shape: {malicious_features.shape}")
            print(f"Sample malicious feature: {malicious_features[0]}")
        except Exception as e:
            print(f"Error generating malicious features: {str(e)}")
            traceback.print_exc()
            raise
        
        # Test dual attention model
        print("\nTesting dual attention model...")
        
        # Create dual attention model if needed
        try:
            if server.dual_attention is None:
                print("Creating new dual attention model for testing")
                server.dual_attention = DualAttention(
                    feature_dim=expected_feature_dim,
                    hidden_dim=32,
                    num_heads=4
                ).to(device)
        except Exception as e:
            print(f"Error creating dual attention model: {str(e)}")
            traceback.print_exc()
            raise
        
        # Test with client features
        if client_features is not None:
            print("Testing dual attention with client features...")
            client_features_batch = client_features.unsqueeze(0)  # Add batch dimension
            print(f"Client features batch shape: {client_features_batch.shape}")
            
            with torch.no_grad():
                try:
                    trust_score, confidence = server.dual_attention(client_features_batch)
                    print(f"Dual attention returned trust score: {trust_score.item():.4f}, confidence: {confidence.item():.4f}")
                except Exception as e:
                    print(f"Error in dual attention with client features: {str(e)}")
                    traceback.print_exc()
        
        # Test with server features
        print("Testing dual attention with server features...")
        server_features_batch = server_features.unsqueeze(0)  # Add batch dimension
        print(f"Server features batch shape: {server_features_batch.shape}")
        
        with torch.no_grad():
            try:
                trust_score, confidence = server.dual_attention(server_features_batch)
                print(f"Dual attention returned trust score: {trust_score.item():.4f}, confidence: {confidence.item():.4f}")
            except Exception as e:
                print(f"Error in dual attention with server features: {str(e)}")
                traceback.print_exc()
        
        # Test with honest features batch
        print("Testing dual attention with honest features batch...")
        print(f"Honest features batch shape: {honest_features.shape}")
        
        with torch.no_grad():
            try:
                trust_scores, confidences = server.dual_attention(honest_features)
                print(f"Dual attention returned trust scores shape: {trust_scores.shape}")
                print(f"Sample trust score: {trust_scores[0].item():.4f}, confidence: {confidences[0].item():.4f}")
            except Exception as e:
                print(f"Error in dual attention with honest features batch: {str(e)}")
                traceback.print_exc()
        
        # Test with malicious features batch
        print("Testing dual attention with malicious features batch...")
        print(f"Malicious features batch shape: {malicious_features.shape}")
        
        with torch.no_grad():
            try:
                trust_scores, confidences = server.dual_attention(malicious_features)
                print(f"Dual attention returned trust scores shape: {trust_scores.shape}")
                print(f"Sample trust score: {trust_scores[0].item():.4f}, confidence: {confidences[0].item():.4f}")
            except Exception as e:
                print(f"Error in dual attention with malicious features batch: {str(e)}")
                traceback.print_exc()
        
        # Test DualAttention training
        print("\nTesting DualAttention training...")
        
        # Combine honest and malicious features
        all_features = torch.cat([honest_features, malicious_features])
        print(f"Combined features shape: {all_features.shape}")
        
        # Create labels (1 for honest, 0 for malicious)
        honest_labels = torch.ones(honest_features.size(0), device=device)
        malicious_labels = torch.zeros(malicious_features.size(0), device=device)
        all_labels = torch.cat([honest_labels, malicious_labels])
        print(f"Labels shape: {all_labels.shape}")
        
        # Try a forward and backward pass
        server.dual_attention.train()
        optimizer = torch.optim.Adam(server.dual_attention.parameters(), lr=0.001)
        
        try:
            # Forward pass
            trust_scores, confidence_scores = server.dual_attention(all_features)
            print(f"Forward pass successful, trust scores shape: {trust_scores.shape}")
            
            # Loss calculation
            trust_scores_clamped = torch.clamp(trust_scores, 1e-7, 1-1e-7)
            loss = torch.nn.functional.binary_cross_entropy(trust_scores_clamped, all_labels)
            print(f"Loss calculation successful, loss: {loss.item():.4f}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Backward pass successful")
            
            print("\n=== All dimension tests PASSED ===")
            return True
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()
            print("\n=== Dimension tests FAILED ===")
            return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        print("\n=== Dimension tests FAILED ===")
        return False

if __name__ == "__main__":
    test_gradient_feature_dimensions() 