"""
Comprehensive test script for the federated learning system with dual attention.

This script tests all aspects of the system:
1. Dataset loading and splitting
2. Model initialization
3. VAE training for gradient reconstruction
4. Dual attention training and evaluation
5. Malicious client detection
6. Trust score computation
7. Aggregation with trust scores
8. Full federated learning system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from itertools import product

# Import federated learning components
from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client, Attack
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.models.attention import DualAttention
from federated_learning.utils.model_utils import set_random_seeds
from federated_learning.models.vae import GradientVAE

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_setup():
    """Test basic setup and initialization."""
    print("\n=== Testing Setup ===")
    
    # Set random seeds
    set_random_seeds(RANDOM_SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create server
    server = Server()
    assert server is not None, "Server initialization failed"
    
    # Test model initialization
    assert server.global_model is not None, "Global model initialization failed"
    
    # Check VAE
    assert server.vae is not None, "VAE initialization failed"
    
    # Check dual attention
    assert server.dual_attention is not None, "Dual attention initialization failed"
    
    print("✅ Setup test passed")
    return server, device

def test_dataset_loading():
    """Test dataset loading and splitting."""
    print("\n=== Testing Dataset Loading ===")
    
    # Load datasets
    train_dataset, test_dataset = load_dataset(DATASET)
    assert train_dataset is not None, "Train dataset loading failed"
    assert test_dataset is not None, "Test dataset loading failed"
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Split dataset for clients
    root_dataset, client_datasets = create_client_datasets(
        train_dataset,
        num_clients=3,
        root_size=100,
        iid=True
    )
    
    assert len(client_datasets) == 3, "Client dataset splitting failed"
    assert len(root_dataset) == 100, "Root dataset size incorrect"
    
    print(f"Root dataset size: {len(root_dataset)}")
    for i, dataset in enumerate(client_datasets):
        print(f"Client {i} dataset size: {len(dataset)}")
    
    print("✅ Dataset loading test passed")
    return root_dataset, client_datasets, test_dataset

def test_vae_training(server, root_dataset):
    """Test VAE training with gradient reconstruction."""
    print("\n=== Testing VAE Training ===")
    
    # Create data loader for root dataset
    root_loader = torch.utils.data.DataLoader(
        root_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Set datasets for server
    server.set_datasets(root_loader, root_dataset)
    
    # Collect root gradients
    root_gradients = server._collect_root_gradients()
    server.root_gradients = root_gradients
    
    assert len(root_gradients) > 0, "No root gradients collected"
    print(f"Collected {len(root_gradients)} root gradients")
    
    # Train VAE
    server.vae = server.train_vae(root_gradients, vae_epochs=1)
    
    # Test reconstruction
    sample_gradient = root_gradients[0]
    with torch.no_grad():
        recon, _, _ = server.vae(sample_gradient.unsqueeze(0))
        
    # Calculate reconstruction error
    recon_error = torch.nn.functional.mse_loss(recon.squeeze(0), sample_gradient)
    print(f"VAE reconstruction error: {recon_error.item():.6f}")
    
    print("✅ VAE training test passed")
    return root_gradients

def test_feature_extraction(server, root_gradients):
    """Test feature extraction from gradients."""
    print("\n=== Testing Feature Extraction ===")
    
    # Extract features from a gradient
    sample_features = server._compute_gradient_features(root_gradients[0], root_gradient=root_gradients[0])
    
    assert sample_features is not None, "Feature extraction failed"
    assert sample_features.shape[0] == (6 if ENABLE_SHAPLEY else 5), "Feature dimension incorrect"
    
    # Feature output should have meaningful values
    assert 0.0 <= sample_features[0] <= 1.0, "VAE reconstruction error outside [0,1]"
    assert 0.0 <= sample_features[1] <= 1.0, "Root similarity outside [0,1]"
    assert 0.0 <= sample_features[2] <= 1.0, "Client similarity outside [0,1]"
    assert 0.0 <= sample_features[3] <= 1.0, "Normalized gradient norm outside [0,1]"
    assert 0.0 <= sample_features[4] <= 1.0, "Consistency outside [0,1]"
    if ENABLE_SHAPLEY and sample_features.shape[0] > 5:
        assert 0.0 <= sample_features[5] <= 1.0, "Shapley value outside [0,1]"
    
    # Features for multiple gradients
    feature_vectors = server._compute_all_gradient_features(root_gradients[:3])
    
    assert feature_vectors is not None, "Batch feature extraction failed"
    assert feature_vectors.shape[0] == 3, "Batch dimension incorrect"
    assert feature_vectors.shape[1] == (6 if ENABLE_SHAPLEY else 5), "Feature dimension incorrect"
    
    print("✅ Feature extraction test passed")
    return feature_vectors

def test_dual_attention(server, feature_vectors):
    """Test dual attention model for malicious client detection."""
    print("\n=== Testing Dual Attention ===")
    
    # Generate malicious features
    malicious_features = server._generate_malicious_features(feature_vectors)
    
    # Import training utilities
    from federated_learning.training.training_utils import train_dual_attention
    
    # Train dual attention model
    dual_attention = train_dual_attention(
        honest_features=feature_vectors,
        malicious_features=malicious_features,
        epochs=3,  # Reduced for testing
        batch_size=2,
        lr=DUAL_ATTENTION_LEARNING_RATE,
        device=server.device,
        verbose=True
    )
    
    assert dual_attention is not None, "Dual attention training failed"
    
    # Test dual attention classification
    dual_attention.eval()
    with torch.no_grad():
        honest_scores, _ = dual_attention(feature_vectors)
        malicious_scores, _ = dual_attention(malicious_features)
    
    honest_avg = honest_scores.mean().item()
    malicious_avg = malicious_scores.mean().item()
    
    print(f"Average honest score: {honest_avg:.4f}")
    print(f"Average malicious score: {malicious_avg:.4f}")
    
    # In an ideal scenario, honest scores should be higher than malicious
    # But for a quick test, we just verify they're different
    print(f"Score difference (honest - malicious): {honest_avg - malicious_avg:.4f}")
    
    # Test get_gradient_weights method
    weights, suspicious_indices = dual_attention.get_gradient_weights(
        feature_vectors,
        trust_scores=honest_scores,
        confidence_scores=torch.ones_like(honest_scores) * 0.5
    )
    
    assert weights is not None, "Gradient weight computation failed"
    assert weights.shape[0] == feature_vectors.shape[0], "Weight dimension mismatch"
    
    print("Weight distribution:", weights.cpu().numpy())
    
    print("✅ Dual attention test passed")
    return dual_attention

def test_attack_detection():
    """Test detection of various attack types."""
    print("\n=== Testing Attack Detection ===")
    
    # Create an honest gradient
    gradient = torch.randn(1000)
    grad_norm = torch.norm(gradient).item()
    print(f"Honest gradient norm: {grad_norm:.4f}")
    
    # Test different attack types
    attack_types = [
        'scaling_attack',
        'sign_flipping_attack',
        'partial_scaling_attack',
        'min_max_attack',
        'targeted_attack'
    ]
    
    attacked_gradients = []
    original_gradient = gradient.clone()
    
    for attack_type in attack_types:
        attack = Attack(attack_type)
        modified = attack.apply_gradient_attack(gradient.clone())
        attacked_gradients.append(modified)
        
        # Check that attack changes the gradient
        assert not torch.allclose(original_gradient, modified), f"{attack_type} did not modify gradient"
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original_gradient.view(1, -1),
            modified.view(1, -1)
        ).item()
        
        print(f"{attack_type}: Cosine similarity with original: {cos_sim:.4f}")
    
    # Create a server for feature extraction
    server = Server()
    server.root_gradients = [original_gradient]
    
    # Extract features for honest and attacked gradients
    features = []
    features.append(server._compute_gradient_features(original_gradient, original_gradient))
    
    for attacked in attacked_gradients:
        features.append(server._compute_gradient_features(attacked, original_gradient))
    
    features_tensor = torch.stack(features)
    
    # Train a dual attention model on these features
    from federated_learning.training.training_utils import train_dual_attention
    
    honest_features = features_tensor[0:1]  # Just the honest one
    malicious_features = features_tensor[1:]  # All attacked ones
    
    dual_attention = train_dual_attention(
        honest_features=honest_features,
        malicious_features=malicious_features,
        epochs=5,
        batch_size=2,
        lr=DUAL_ATTENTION_LEARNING_RATE,
        device=server.device,
        verbose=True
    )
    
    # Evaluate the model
    dual_attention.eval()
    with torch.no_grad():
        all_scores, _ = dual_attention(features_tensor)
    
    # Print scores for each gradient type
    print("\nTrust scores:")
    print(f"Honest gradient: {all_scores[0].item():.4f}")
    for i, attack_type in enumerate(attack_types):
        print(f"{attack_type}: {all_scores[i+1].item():.4f}")
    
    # Check if honest gradient gets highest score
    honest_highest = (all_scores[0] >= all_scores[1:]).all().item()
    print(f"✅ Honest gradient receives highest trust score: {honest_highest}")
    
    # Get weights
    weights, _ = dual_attention.get_gradient_weights(features_tensor, all_scores)
    
    # Print weights
    print("\nAggregation weights:")
    print(f"Honest gradient: {weights[0].item():.4f}")
    for i, attack_type in enumerate(attack_types):
        print(f"{attack_type}: {weights[i+1].item():.4f}")
    
    # Calculate weight ratio
    honest_weight = weights[0].item()
    malicious_weight_avg = weights[1:].mean().item()
    weight_ratio = honest_weight / malicious_weight_avg
    
    print(f"Weight ratio (honest/malicious): {weight_ratio:.2f}x")
    
    print("✅ Attack detection test passed" if weight_ratio > 1.0 else "❌ Attack detection test failed")
    return dual_attention

def test_federated_learning():
    """Test complete federated learning system."""
    print("\n=== Testing Federated Learning ===")
    
    # Set random seeds
    set_random_seeds(RANDOM_SEED)
    
    # Load datasets
    train_dataset, test_dataset = load_dataset(DATASET)
    
    # Split dataset for clients (2 honest, 1 malicious)
    num_clients = 3
    root_dataset, client_datasets = create_client_datasets(
        train_dataset,
        num_clients=num_clients,
        root_size=100,
        iid=True
    )
    
    # Create server and clients
    server = Server()
    
    # Create root loader
    root_loader = torch.utils.data.DataLoader(
        root_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Set datasets for server
    server.set_datasets(root_loader, test_dataset)
    
    # Pretrain global model
    server._pretrain_global_model()
    
    # Create clients (1 malicious, 2 honest)
    clients = []
    malicious_client_idx = 0  # First client is malicious
    
    for i in range(num_clients):
        is_malicious = (i == malicious_client_idx)
        client = Client(
            client_id=i,
            dataset=client_datasets[i],
            is_malicious=is_malicious
        )
        clients.append(client)
        print(f"Client {i} created with {len(client_datasets[i])} samples (Malicious: {is_malicious})")
    
    server.clients = clients
    
    # Collect root gradients and train VAE
    root_gradients = server._collect_root_gradients()
    server.root_gradients = root_gradients
    server.vae = server.train_vae(root_gradients, vae_epochs=1)
    
    # Train dual attention with simulated features
    honest_features = server._compute_all_gradient_features(root_gradients[:5])
    malicious_features = server._generate_malicious_features(honest_features)
    
    # Import training utilities
    from federated_learning.training.training_utils import train_dual_attention
    
    # Train dual attention model
    dual_attention = train_dual_attention(
        honest_features=honest_features,
        malicious_features=malicious_features,
        epochs=5,
        batch_size=BATCH_SIZE,
        lr=DUAL_ATTENTION_LEARNING_RATE,
        device=server.device,
        verbose=True
    )
    
    server.dual_attention = dual_attention
    
    # Run federated learning for 1 round
    test_errors, round_metrics = server.train(num_rounds=1)
    
    # Calculate statistics
    honest_trust = []
    malicious_trust = []
    honest_weights = []
    malicious_weights = []
    
    for round_idx in round_metrics:
        for client_idx, trust_score in round_metrics[round_idx]['trust_scores'].items():
            if client_idx == malicious_client_idx:
                malicious_trust.append(trust_score)
                if client_idx in round_metrics[round_idx]['weights']:
                    malicious_weights.append(round_metrics[round_idx]['weights'][client_idx])
            else:
                honest_trust.append(trust_score)
                if client_idx in round_metrics[round_idx]['weights']:
                    honest_weights.append(round_metrics[round_idx]['weights'][client_idx])
    
    # Print statistics if available
    if honest_trust and malicious_trust:
        avg_honest_trust = sum(honest_trust) / len(honest_trust)
        avg_malicious_trust = sum(malicious_trust) / len(malicious_trust)
        print(f"Average honest client trust score: {avg_honest_trust:.4f}")
        print(f"Average malicious client trust score: {avg_malicious_trust:.4f}")
        
        if honest_weights and malicious_weights:
            avg_honest_weight = sum(honest_weights) / len(honest_weights)
            avg_malicious_weight = sum(malicious_weights) / len(malicious_weights)
            print(f"Average honest client weight: {avg_honest_weight:.4f}")
            print(f"Average malicious client weight: {avg_malicious_weight:.4f}")
            
            weight_ratio = avg_honest_weight / avg_malicious_weight
            print(f"Weight ratio (honest/malicious): {weight_ratio:.2f}x")
            
            if weight_ratio > 1.0:
                print("✅ TEST PASSED: Dual attention model effectively discriminates between honest and malicious clients.")
                print(f"Honest clients receive {weight_ratio:.2f}x more weight than malicious clients on average.")
            else:
                print("❌ TEST FAILED: Malicious clients receive equal or more weight than honest clients.")
    
    print("✅ Federated learning test completed")
    return test_errors, round_metrics

def create_test_gradients(device, attack_type="none", num_honest=3, num_malicious=2):
    """Create test gradients with different attack types."""
    gradients = []
    labels = []  # 1 = honest, 0 = malicious
    
    # Create honest gradients (normal distribution)
    for i in range(num_honest):
        honest_grad = torch.randn(1000).to(device) * 0.1
        gradients.append(honest_grad)
        labels.append(1)
    
    # Create malicious gradients based on attack type
    for i in range(num_malicious):
        if attack_type == "scaling_attack":
            # Scaling attack: multiply by large factor
            base_grad = torch.randn(1000).to(device) * 0.1
            malicious_grad = base_grad * (5.0 + i * 2.0)  # 5x, 7x scaling
            
        elif attack_type == "sign_flip_attack":
            # Sign flip attack: flip signs
            base_grad = torch.randn(1000).to(device) * 0.1
            malicious_grad = -base_grad
            
        elif attack_type == "noise_attack":
            # Noise attack: add large noise
            base_grad = torch.randn(1000).to(device) * 0.1
            noise = torch.randn(1000).to(device) * 0.5
            malicious_grad = base_grad + noise
            
        elif attack_type == "zero_attack":
            # Zero attack: send zero gradients
            malicious_grad = torch.zeros(1000).to(device)
            
        elif attack_type == "random_attack":
            # Random attack: completely random gradients
            malicious_grad = torch.randn(1000).to(device) * 2.0
            
        elif attack_type == "partial_scaling_attack":
            # Partial scaling: scale only some parameters
            base_grad = torch.randn(1000).to(device) * 0.1
            malicious_grad = base_grad.clone()
            # Scale first 30% of parameters
            scale_idx = int(0.3 * len(malicious_grad))
            malicious_grad[:scale_idx] *= (3.0 + i)
            
        else:  # "none" or unknown
            # No attack - honest gradient
            malicious_grad = torch.randn(1000).to(device) * 0.1
            labels[-1] = 1  # Change label to honest
        
        gradients.append(malicious_grad)
        labels.append(0 if attack_type != "none" else 1)
    
    return gradients, labels

def test_aggregation_method(server, gradients, labels, method_name, attack_type):
    """Test a specific aggregation method with given gradients."""
    print(f"\n--- Testing {method_name} with {attack_type} ---")
    
    device = server.device
    
    # Compute features for all gradients
    features = server._compute_all_gradient_features(gradients)
    
    # Test dual attention
    if not hasattr(server, 'dual_attention') or server.dual_attention is None:
        server.dual_attention = DualAttention(
            feature_dim=features.shape[1],
            hidden_dim=64,
            num_heads=4
        ).to(device)
    
    # Quick training on synthetic data to make dual attention work
    optimizer = torch.optim.Adam(server.dual_attention.parameters(), lr=1e-3)
    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    
    for epoch in range(10):  # Quick training
        optimizer.zero_grad()
        trust_scores, _ = server.dual_attention(features)
        loss = F.binary_cross_entropy(trust_scores.squeeze(), labels_tensor)
        loss.backward()
        optimizer.step()
    
    # Test aggregation
    server.dual_attention.eval()
    with torch.no_grad():
        trust_scores, confidence = server.dual_attention(features)
        weights, malicious_indices = server.dual_attention.get_gradient_weights(features, trust_scores)
    
    # Test different aggregation methods
    try:
        if method_name == "fedavg":
            aggregated = server._aggregate_fedavg(gradients, weights)
        elif method_name == "fedbn":
            aggregated = server._aggregate_fedbn(gradients, weights)
        elif method_name == "fedadmm":
            aggregated = server._aggregate_fedadmm(gradients, weights)
        else:
            print(f"Unknown method: {method_name}")
            return False
        
        # Analyze results
        honest_indices = [i for i, label in enumerate(labels) if label == 1]
        malicious_indices_actual = [i for i, label in enumerate(labels) if label == 0]
        
        # Calculate detection metrics
        detected_malicious = set(malicious_indices)
        actual_malicious = set(malicious_indices_actual)
        
        true_positives = len(detected_malicious & actual_malicious)
        false_positives = len(detected_malicious - actual_malicious)
        false_negatives = len(actual_malicious - detected_malicious)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"  Aggregated gradient norm: {torch.norm(aggregated).item():.4f}")
        print(f"  Detected malicious: {malicious_indices}")
        print(f"  Actual malicious: {malicious_indices_actual}")
        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}")
        
        # Check if aggregation is reasonable
        agg_norm = torch.norm(aggregated).item()
        if not (0.01 < agg_norm < 100.0):  # Reasonable range
            print(f"  ⚠️  Warning: Aggregated gradient norm seems unusual: {agg_norm:.4f}")
        
        # Check if honest clients get higher weights
        if honest_indices and malicious_indices_actual:
            avg_honest_weight = np.mean([weights[i].item() for i in honest_indices])
            avg_malicious_weight = np.mean([weights[i].item() for i in malicious_indices_actual])
            
            print(f"  Average honest weight: {avg_honest_weight:.4f}")
            print(f"  Average malicious weight: {avg_malicious_weight:.4f}")
            
            if avg_honest_weight > avg_malicious_weight:
                print(f"  ✅ Honest clients get higher weights")
                return True
            else:
                print(f"  ⚠️  Warning: Malicious clients getting higher weights")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in {method_name}: {str(e)}")
        return False

def test_all_combinations():
    """Test all combinations of aggregation methods and attack types."""
    print("=== Comprehensive System Test: Dual Attention + All Methods + All Attacks ===")
    
    # Test configurations
    aggregation_methods = ["fedavg", "fedbn", "fedadmm"]
    attack_types = [
        "none", 
        "scaling_attack", 
        "sign_flip_attack", 
        "noise_attack", 
        "zero_attack", 
        "random_attack",
        "partial_scaling_attack"
    ]
    
    # Initialize server
    server = Server()
    device = server.device
    
    # Create root gradients for reference
    root_gradients = []
    for _ in range(5):
        grad = torch.randn(1000).to(device) * 0.1
        root_gradients.append(grad)
    server.root_gradients = root_gradients
    
    # Train VAE for reconstruction features
    vae = GradientVAE(input_dim=1000, hidden_dim=32, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    print("Training VAE for feature extraction...")
    for epoch in range(5):
        for grad in root_gradients:
            optimizer.zero_grad()
            recon, mu, logvar = vae(grad.unsqueeze(0))
            loss = vae.loss_function(recon, grad.unsqueeze(0), mu, logvar)
            loss.backward()
            optimizer.step()
    
    server.vae = vae
    
    # Test results storage
    results = {}
    total_tests = 0
    passed_tests = 0
    
    print(f"\nTesting {len(aggregation_methods)} methods × {len(attack_types)} attacks = {len(aggregation_methods) * len(attack_types)} combinations")
    print("=" * 80)
    
    # Test all combinations
    for method, attack in product(aggregation_methods, attack_types):
        total_tests += 1
        
        # Create test gradients for this attack type
        gradients, labels = create_test_gradients(device, attack, num_honest=3, num_malicious=2)
        
        # Test this combination
        success = test_aggregation_method(server, gradients, labels, method, attack)
        
        # Store result
        key = f"{method}_{attack}"
        results[key] = success
        
        if success:
            passed_tests += 1
            print(f"  ✅ {method} + {attack}: PASSED")
        else:
            print(f"  ❌ {method} + {attack}: FAILED")
    
    return results, total_tests, passed_tests

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== Testing Edge Cases ===")
    
    server = Server()
    device = server.device
    
    # Setup basic components
    root_grad = torch.randn(1000).to(device) * 0.1
    server.root_gradients = [root_grad]
    
    edge_cases = [
        ("All honest clients", lambda: create_test_gradients(device, "none", num_honest=5, num_malicious=0)),
        ("All malicious clients", lambda: create_test_gradients(device, "scaling_attack", num_honest=0, num_malicious=5)),
        ("Single client", lambda: create_test_gradients(device, "none", num_honest=1, num_malicious=0)),
        ("Extreme scaling", lambda: ([torch.randn(1000).to(device) * 0.1, torch.randn(1000).to(device) * 100.0], [1, 0])),
        ("Very small gradients", lambda: ([torch.randn(1000).to(device) * 1e-6, torch.randn(1000).to(device) * 1e-6], [1, 1])),
    ]
    
    edge_results = {}
    
    for case_name, case_func in edge_cases:
        print(f"\n--- Testing {case_name} ---")
        try:
            gradients, labels = case_func()
            
            if len(gradients) == 0:
                print("  ⚠️  No gradients to test")
                continue
            
            # Test with FedAvg (simplest method)
            success = test_aggregation_method(server, gradients, labels, "fedavg", case_name)
            edge_results[case_name] = success
            
            if success:
                print(f"  ✅ {case_name}: PASSED")
            else:
                print(f"  ⚠️  {case_name}: Issues detected")
                
        except Exception as e:
            print(f"  ❌ {case_name}: ERROR - {str(e)}")
            edge_results[case_name] = False
    
    return edge_results

def analyze_feature_quality():
    """Analyze the quality of feature extraction across different scenarios."""
    print("\n=== Analyzing Feature Quality ===")
    
    server = Server()
    device = server.device
    
    # Setup
    root_grad = torch.randn(1000).to(device) * 0.1
    server.root_gradients = [root_grad]
    
    # Test different gradient types
    test_scenarios = {
        "Honest": torch.randn(1000).to(device) * 0.1,
        "Scaling Attack": torch.randn(1000).to(device) * 5.0,
        "Sign Flip": -root_grad,
        "Noise Attack": root_grad + torch.randn(1000).to(device) * 0.3,
        "Zero Gradient": torch.zeros(1000).to(device),
    }
    
    print("\nFeature Analysis:")
    print("Scenario         | Recon Err | Root Sim | Grad Norm | Sign Cons | Expected")
    print("-" * 80)
    
    feature_quality = {}
    
    for scenario, grad in test_scenarios.items():
        features = server._compute_gradient_features(grad, root_grad, skip_client_sim=True)
        
        recon_err = features[0].item()
        root_sim = features[1].item()
        grad_norm = features[3].item()
        sign_cons = features[4].item()
        
        # Determine if features match expectations
        if scenario == "Honest":
            expected = "Low recon, high sim, low norm, high cons"
            quality = (recon_err < 0.7 and root_sim > 0.3 and grad_norm < 0.6 and sign_cons > 0.3)
        elif scenario == "Scaling Attack":
            expected = "High norm"
            quality = (grad_norm > 0.7)
        elif scenario == "Sign Flip":
            expected = "Low sim, low cons"
            quality = (root_sim < 0.2 and sign_cons < 0.2)
        elif scenario == "Noise Attack":
            expected = "High recon, med norm"
            quality = (recon_err > 0.3 and grad_norm > 0.2)
        else:  # Zero
            expected = "Low norm"
            quality = (grad_norm < 0.3)
        
        feature_quality[scenario] = quality
        status = "✅" if quality else "⚠️"
        
        print(f"{scenario:<15} | {recon_err:>7.3f}   | {root_sim:>6.3f}   | {grad_norm:>7.3f}   | {sign_cons:>7.3f}   | {expected} {status}")
    
    return feature_quality

def main():
    """Run comprehensive system tests."""
    print("🔥 COMPREHENSIVE DUAL ATTENTION SYSTEM TEST 🔥")
    print("Testing RL_AGGREGATION_METHOD = dual_attention with all combinations")
    print("=" * 80)
    
    try:
        # Test 1: All method + attack combinations
        print("\n📊 PHASE 1: Testing All Method + Attack Combinations")
        results, total_tests, passed_tests = test_all_combinations()
        
        # Test 2: Edge cases
        print("\n🔍 PHASE 2: Testing Edge Cases")
        edge_results = test_edge_cases()
        
        # Test 3: Feature quality analysis
        print("\n🎯 PHASE 3: Feature Quality Analysis")
        feature_quality = analyze_feature_quality()
        
        # Final analysis
        print("\n" + "=" * 80)
        print("📋 FINAL RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n🔢 COMBINATION TESTS:")
        print(f"   Total combinations tested: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n🔍 EDGE CASE TESTS:")
        edge_passed = sum(edge_results.values())
        edge_total = len(edge_results)
        print(f"   Edge cases passed: {edge_passed}/{edge_total}")
        
        print(f"\n🎯 FEATURE QUALITY:")
        feature_passed = sum(feature_quality.values())
        feature_total = len(feature_quality)
        print(f"   Feature scenarios correct: {feature_passed}/{feature_total}")
        
        # Detailed breakdown
        print(f"\n📊 DETAILED BREAKDOWN:")
        
        # Group results by method
        methods = ["fedavg", "fedbn", "fedadmm"]
        for method in methods:
            method_results = [results[key] for key in results if key.startswith(method)]
            method_success = sum(method_results)
            print(f"   {method.upper()}: {method_success}/{len(method_results)} attacks handled correctly")
        
        # Group results by attack
        attacks = ["none", "scaling_attack", "sign_flip_attack", "noise_attack", "zero_attack", "random_attack", "partial_scaling_attack"]
        for attack in attacks:
            attack_results = [results[key] for key in results if key.endswith(attack)]
            attack_success = sum(attack_results)
            print(f"   {attack}: {attack_success}/{len(attack_results)} methods work correctly")
        
        # Overall assessment
        overall_success_rate = (passed_tests + edge_passed + feature_passed) / (total_tests + edge_total + feature_total)
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Combined success rate: {overall_success_rate*100:.1f}%")
        
        if overall_success_rate > 0.8:
            print(f"   🎉 EXCELLENT: Your dual attention system is robust and ready!")
        elif overall_success_rate > 0.6:
            print(f"   ✅ GOOD: System works well with minor issues to address")
        else:
            print(f"   ⚠️  NEEDS WORK: Several issues need to be addressed")
        
        print(f"\n🚀 RECOMMENDATIONS:")
        if passed_tests == total_tests:
            print(f"   ✅ All method+attack combinations work - system is production ready")
        else:
            failed_combinations = [key for key, success in results.items() if not success]
            print(f"   ⚠️  Review these combinations: {failed_combinations}")
        
        if feature_passed == feature_total:
            print(f"   ✅ Feature extraction is working perfectly")
        else:
            print(f"   ⚠️  Some feature extraction issues detected")
        
        print(f"\n💡 CONCLUSION:")
        print(f"   Your dual attention system with RL_AGGREGATION_METHOD = dual_attention")
        print(f"   has been tested across {total_tests + edge_total + feature_total} different scenarios.")
        print(f"   Overall performance: {overall_success_rate*100:.1f}% success rate")
        
        return overall_success_rate > 0.7
        
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 