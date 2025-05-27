"""
Test script to verify zero attack detection fix.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.models.attention import DualAttention
from federated_learning.models.vae import GradientVAE
from federated_learning.utils.model_utils import set_random_seeds

def test_zero_attack_detection():
    """Test that zero attacks are properly detected and penalized."""
    print("🔍 Testing Zero Attack Detection Fix")
    print("=" * 50)
    
    # Set random seeds
    set_random_seeds(RANDOM_SEED)
    
    # Create server
    server = Server()
    device = server.device
    
    # Create test gradients: 3 honest, 2 zero attacks
    gradients = []
    labels = []  # 1 = honest, 0 = malicious
    
    # Honest gradients (normal distribution)
    for i in range(3):
        honest_grad = torch.randn(1000).to(device) * 0.1
        gradients.append(honest_grad)
        labels.append(1)
        print(f"Honest gradient {i}: norm = {torch.norm(honest_grad).item():.4f}")
    
    # Zero attack gradients
    for i in range(2):
        zero_grad = torch.zeros(1000).to(device)
        gradients.append(zero_grad)
        labels.append(0)
        print(f"Zero attack {i}: norm = {torch.norm(zero_grad).item():.4f}")
    
    # Set up server with root gradients
    server.root_gradients = gradients[:1]  # Use first honest gradient as root
    
    # Train a simple VAE for feature extraction
    vae = GradientVAE(input_dim=1000, hidden_dim=32, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    print("\nTraining VAE...")
    for epoch in range(5):
        for grad in gradients[:3]:  # Train on honest gradients only
            optimizer.zero_grad()
            recon, mu, logvar = vae(grad.unsqueeze(0))
            loss = vae.loss_function(recon, grad.unsqueeze(0), mu, logvar)
            loss.backward()
            optimizer.step()
    
    server.vae = vae
    
    # Extract features for all gradients
    print("\nExtracting features...")
    features = server._compute_all_gradient_features(gradients)
    
    print(f"\nFeature analysis:")
    print("Client | Type    | Recon Err | Root Sim | Grad Norm | Sign Cons")
    print("-" * 60)
    
    for i, (grad, label) in enumerate(zip(gradients, labels)):
        client_type = "Honest" if label == 1 else "Zero Attack"
        recon_err = features[i, 0].item()
        root_sim = features[i, 1].item()
        grad_norm = features[i, 3].item()
        sign_cons = features[i, 4].item()
        
        print(f"{i:6d} | {client_type:8s} | {recon_err:7.3f}   | {root_sim:6.3f}   | {grad_norm:7.3f}   | {sign_cons:7.3f}")
    
    # Create dual attention model
    dual_attention = DualAttention(
        feature_dim=features.shape[1],
        hidden_dim=64,
        num_heads=4
    ).to(device)
    
    # Quick training to make dual attention functional
    optimizer = torch.optim.Adam(dual_attention.parameters(), lr=1e-3)
    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    
    print("\nTraining dual attention...")
    for epoch in range(20):
        optimizer.zero_grad()
        malicious_scores, _ = dual_attention(features)
        loss = F.binary_cross_entropy(malicious_scores, labels_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Test the fixed weight calculation
    print("\n" + "=" * 50)
    print("🎯 Testing Weight Calculation with Zero Attack Fix")
    print("=" * 50)
    
    dual_attention.eval()
    with torch.no_grad():
        malicious_scores, confidence = dual_attention(features)
        weights, malicious_indices = dual_attention.get_gradient_weights(features, malicious_scores)
    
    # Analyze results
    print(f"\n📊 Results Analysis:")
    print("Client | Type        | Malicious Score | Weight  | Detected")
    print("-" * 55)
    
    honest_weights = []
    zero_weights = []
    
    for i, (label, score, weight) in enumerate(zip(labels, malicious_scores, weights)):
        client_type = "Honest" if label == 1 else "Zero Attack"
        detected = "YES" if i in malicious_indices else "NO"
        
        print(f"{i:6d} | {client_type:11s} | {score.item():13.4f} | {weight.item():6.4f} | {detected:8s}")
        
        if label == 1:
            honest_weights.append(weight.item())
        else:
            zero_weights.append(weight.item())
    
    # Calculate metrics
    print(f"\n📈 Performance Metrics:")
    
    # Detection accuracy
    detected_malicious = set(malicious_indices)
    actual_malicious = set([i for i, label in enumerate(labels) if label == 0])
    
    true_positives = len(detected_malicious & actual_malicious)
    false_positives = len(detected_malicious - actual_malicious)
    false_negatives = len(actual_malicious - detected_malicious)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Detection Precision: {precision:.2f}")
    print(f"Detection Recall: {recall:.2f}")
    print(f"Detection F1-Score: {f1_score:.2f}")
    
    # Weight analysis
    if honest_weights and zero_weights:
        avg_honest_weight = np.mean(honest_weights)
        avg_zero_weight = np.mean(zero_weights)
        weight_ratio = avg_honest_weight / avg_zero_weight if avg_zero_weight > 0 else float('inf')
        
        print(f"Average honest weight: {avg_honest_weight:.4f}")
        print(f"Average zero attack weight: {avg_zero_weight:.4f}")
        print(f"Weight ratio (honest/zero): {weight_ratio:.2f}x")
        
        # Success criteria
        detection_success = recall >= 0.8  # At least 80% of zero attacks detected
        weight_success = weight_ratio >= 2.0  # Honest clients get at least 2x more weight
        
        print(f"\n🎯 Test Results:")
        print(f"Zero Attack Detection: {'✅ PASS' if detection_success else '❌ FAIL'} (Recall: {recall:.2f})")
        print(f"Weight Discrimination: {'✅ PASS' if weight_success else '❌ FAIL'} (Ratio: {weight_ratio:.2f}x)")
        
        overall_success = detection_success and weight_success
        print(f"Overall Test: {'✅ PASS' if overall_success else '❌ FAIL'}")
        
        return overall_success
    else:
        print("❌ FAIL: Could not calculate weight metrics")
        return False

def test_aggregation_with_zero_attacks():
    """Test that aggregation methods work correctly with zero attack detection."""
    print("\n" + "=" * 50)
    print("🔧 Testing Aggregation with Zero Attack Detection")
    print("=" * 50)
    
    # Set random seeds
    set_random_seeds(RANDOM_SEED)
    
    # Create server
    server = Server()
    device = server.device
    
    # Create test gradients: 2 honest, 3 zero attacks
    gradients = []
    labels = []
    
    # Honest gradients
    for i in range(2):
        honest_grad = torch.randn(1000).to(device) * 0.1
        gradients.append(honest_grad)
        labels.append(1)
    
    # Zero attack gradients
    for i in range(3):
        zero_grad = torch.zeros(1000).to(device)
        gradients.append(zero_grad)
        labels.append(0)
    
    # Set up server
    server.root_gradients = gradients[:1]
    
    # Simple VAE setup
    vae = GradientVAE(input_dim=1000, hidden_dim=32, latent_dim=16).to(device)
    server.vae = vae
    
    # Extract features
    features = server._compute_all_gradient_features(gradients)
    
    # Create and setup dual attention
    dual_attention = DualAttention(
        feature_dim=features.shape[1],
        hidden_dim=64,
        num_heads=4
    ).to(device)
    
    server.dual_attention = dual_attention
    
    # Test different aggregation methods
    methods = ['fedavg', 'fedbn', 'fedadmm']
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} with zero attacks ---")
        
        try:
            # Get weights using dual attention
            with torch.no_grad():
                malicious_scores, confidence = dual_attention(features)
                weights, malicious_indices = dual_attention.get_gradient_weights(features, malicious_scores)
            
            # Test aggregation
            if method == 'fedavg':
                aggregated = server._aggregate_fedavg(gradients, weights)
            elif method == 'fedbn':
                aggregated = server._aggregate_fedbn(gradients, weights)
            elif method == 'fedadmm':
                aggregated = server._aggregate_fedadmm(gradients, weights)
            
            # Analyze results
            agg_norm = torch.norm(aggregated).item()
            
            # Check if zero attacks were detected
            zero_indices = [i for i, label in enumerate(labels) if label == 0]
            detected_zeros = len(set(malicious_indices) & set(zero_indices))
            
            print(f"Aggregated gradient norm: {agg_norm:.4f}")
            print(f"Zero attacks detected: {detected_zeros}/{len(zero_indices)}")
            
            # Success if most zero attacks detected and aggregation is reasonable
            detection_rate = detected_zeros / len(zero_indices)
            success = detection_rate >= 0.6 and 0.01 < agg_norm < 10.0
            
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    print("🚀 Zero Attack Detection Fix Test")
    print("=" * 50)
    
    try:
        # Test 1: Basic zero attack detection
        success1 = test_zero_attack_detection()
        
        # Test 2: Aggregation with zero attacks
        test_aggregation_with_zero_attacks()
        
        print("\n" + "=" * 50)
        print("🏁 Final Results")
        print("=" * 50)
        
        if success1:
            print("✅ Zero attack detection fix is working correctly!")
            print("✅ The system now properly detects and penalizes zero gradient attacks.")
            print("✅ Honest clients receive significantly higher weights than zero attackers.")
        else:
            print("❌ Zero attack detection fix needs more work.")
            print("❌ The system may still have issues with zero gradient detection.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 