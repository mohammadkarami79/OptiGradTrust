"""
Comprehensive Feature Calculation and Dual Attention Validation Tests
This script tests each feature individually and validates the dual attention mechanism.
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
from federated_learning.models.vae import GradientVAE
from federated_learning.models.attention import DualAttention

def test_individual_features():
    """Test each feature calculation individually."""
    print("\n=== Testing Individual Feature Calculations ===")
    
    server = Server()
    device = server.device
    
    # Create controlled test gradients
    root_grad = torch.randn(1000).to(device) * 0.1
    server.root_gradients = [root_grad]
    
    # Train a simple VAE
    vae = GradientVAE(input_dim=1000, hidden_dim=32, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    for epoch in range(5):
        optimizer.zero_grad()
        recon, mu, logvar = vae(root_grad.unsqueeze(0))
        loss = vae.loss_function(recon, root_grad.unsqueeze(0), mu, logvar)
        loss.backward()
        optimizer.step()
    
    server.vae = vae
    
    print("\n--- Test 1: VAE Reconstruction Error ---")
    # Test identical gradient (should have low reconstruction error)
    identical_grad = root_grad.clone()
    features_identical = server._compute_gradient_features(identical_grad, root_grad)
    recon_error_identical = features_identical[0].item()
    
    # Test very different gradient (should have high reconstruction error)
    different_grad = torch.randn(1000).to(device) * 5.0  # Much larger and different
    features_different = server._compute_gradient_features(different_grad, root_grad)
    recon_error_different = features_different[0].item()
    
    print(f"Identical gradient reconstruction error: {recon_error_identical:.4f}")
    print(f"Different gradient reconstruction error: {recon_error_different:.4f}")
    
    assert recon_error_different > recon_error_identical, "Different gradient should have higher reconstruction error"
    print("✅ VAE reconstruction error test passed")
    
    print("\n--- Test 2: Root Similarity ---")
    # Test identical gradient (should have similarity = 1.0)
    root_sim_identical = features_identical[1].item()
    
    # Test opposite gradient (should have similarity = 0.0)
    opposite_grad = -root_grad
    features_opposite = server._compute_gradient_features(opposite_grad, root_grad)
    root_sim_opposite = features_opposite[1].item()
    
    # Test orthogonal gradient (should have similarity ≈ 0.5)
    orthogonal_grad = torch.randn(1000).to(device) * 0.1
    # Make it orthogonal to root_grad
    orthogonal_grad = orthogonal_grad - torch.dot(orthogonal_grad, root_grad) / torch.dot(root_grad, root_grad) * root_grad
    features_orthogonal = server._compute_gradient_features(orthogonal_grad, root_grad)
    root_sim_orthogonal = features_orthogonal[1].item()
    
    print(f"Identical gradient root similarity: {root_sim_identical:.4f}")
    print(f"Opposite gradient root similarity: {root_sim_opposite:.4f}")
    print(f"Orthogonal gradient root similarity: {root_sim_orthogonal:.4f}")
    
    assert root_sim_identical > 0.95, "Identical gradient should have very high similarity"
    assert root_sim_opposite < 0.05, "Opposite gradient should have very low similarity"
    assert 0.4 < root_sim_orthogonal < 0.6, "Orthogonal gradient should have medium similarity"
    print("✅ Root similarity test passed")
    
    print("\n--- Test 3: Gradient Norm Feature ---")
    # Test small gradient
    small_grad = torch.randn(1000).to(device) * 0.01
    features_small = server._compute_gradient_features(small_grad, root_grad)
    norm_feature_small = features_small[3].item()
    
    # Test large gradient
    large_grad = torch.randn(1000).to(device) * 10.0
    features_large = server._compute_gradient_features(large_grad, root_grad)
    norm_feature_large = features_large[3].item()
    
    print(f"Small gradient ({torch.norm(small_grad).item():.4f}) norm feature: {norm_feature_small:.4f}")
    print(f"Large gradient ({torch.norm(large_grad).item():.4f}) norm feature: {norm_feature_large:.4f}")
    
    assert norm_feature_large > norm_feature_small, "Large gradient should have higher norm feature"
    print("✅ Gradient norm feature test passed")
    
    print("\n--- Test 4: Sign Consistency ---")
    # Test identical signs (should be 1.0)
    same_sign_grad = torch.abs(root_grad) * torch.sign(root_grad)
    features_same_sign = server._compute_gradient_features(same_sign_grad, root_grad)
    sign_consistency_same = features_same_sign[4].item()
    
    # Test opposite signs (should be 0.0)
    opposite_sign_grad = -torch.abs(root_grad) * torch.sign(root_grad)
    features_opposite_sign = server._compute_gradient_features(opposite_sign_grad, root_grad)
    sign_consistency_opposite = features_opposite_sign[4].item()
    
    print(f"Same sign gradient sign consistency: {sign_consistency_same:.4f}")
    print(f"Opposite sign gradient sign consistency: {sign_consistency_opposite:.4f}")
    
    assert sign_consistency_same > 0.95, "Same sign gradient should have high consistency"
    assert sign_consistency_opposite < 0.05, "Opposite sign gradient should have low consistency"
    print("✅ Sign consistency test passed")
    
    return server

def test_attack_detection_accuracy():
    """Test the accuracy of attack detection."""
    print("\n=== Testing Attack Detection Accuracy ===")
    
    server = Server()
    device = server.device
    
    # Create root gradients
    root_gradients = []
    for _ in range(5):
        grad = torch.randn(1000).to(device) * 0.1
        root_gradients.append(grad)
    server.root_gradients = root_gradients
    
    # Train VAE
    vae = GradientVAE(input_dim=1000, hidden_dim=32, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    for epoch in range(5):
        for grad in root_gradients:
            optimizer.zero_grad()
            recon, mu, logvar = vae(grad.unsqueeze(0))
            loss = vae.loss_function(recon, grad.unsqueeze(0), mu, logvar)
            loss.backward()
            optimizer.step()
    
    server.vae = vae
    
    # Create test scenarios
    test_cases = []
    labels = []
    
    # Honest gradients (label = 1)
    for _ in range(10):
        honest_grad = torch.randn(1000).to(device) * 0.1
        test_cases.append(honest_grad)
        labels.append(1)
    
    # Scaling attacks (label = 0)
    for _ in range(5):
        scaling_attack = torch.randn(1000).to(device) * 2.0
        test_cases.append(scaling_attack)
        labels.append(0)
    
    # Sign flip attacks (label = 0)
    for _ in range(5):
        sign_flip = -root_gradients[0] + torch.randn(1000).to(device) * 0.01
        test_cases.append(sign_flip)
        labels.append(0)
    
    # Compute features for all test cases
    features = server._compute_all_gradient_features(test_cases)
    
    # Create and train dual attention
    dual_attention = DualAttention(
        feature_dim=features.shape[1],
        hidden_dim=64,
        num_heads=4
    ).to(device)
    
    # Simple training
    optimizer = torch.optim.Adam(dual_attention.parameters(), lr=1e-3)
    labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    
    for epoch in range(20):
        optimizer.zero_grad()
        trust_scores, _ = dual_attention(features)
        loss = F.binary_cross_entropy(trust_scores.squeeze(), labels_tensor)
        loss.backward()
        optimizer.step()
    
    # Test detection accuracy
    dual_attention.eval()
    with torch.no_grad():
        trust_scores, _ = dual_attention(features)
        weights, malicious_indices = dual_attention.get_gradient_weights(features, trust_scores)
        
        # Convert trust scores to binary predictions
        predictions = (trust_scores.squeeze() > 0.5).cpu().numpy().astype(int)
        
        # Calculate accuracy
        correct = np.sum(predictions == np.array(labels))
        accuracy = correct / len(labels)
        
        print(f"Detection accuracy: {accuracy:.2%} ({correct}/{len(labels)})")
        print(f"Detected {len(malicious_indices)} malicious clients out of {np.sum(np.array(labels) == 0)} actual malicious")
        
        # Check if we detect most attacks
        assert accuracy > 0.7, f"Detection accuracy should be > 70%, got {accuracy:.2%}"
        
    print("✅ Attack detection accuracy test passed")
    return accuracy

def test_feature_robustness():
    """Test feature calculation robustness to noise and edge cases."""
    print("\n=== Testing Feature Robustness ===")
    
    server = Server()
    device = server.device
    
    # Create root gradient
    root_grad = torch.randn(1000).to(device) * 0.1
    server.root_gradients = [root_grad]
    
    # Train VAE
    vae = GradientVAE(input_dim=1000, hidden_dim=32, latent_dim=16).to(device)
    server.vae = vae
    
    print("\n--- Test 1: Zero Gradient ---")
    zero_grad = torch.zeros(1000).to(device)
    features_zero = server._compute_gradient_features(zero_grad, root_grad)
    print(f"Zero gradient features: {features_zero.cpu().numpy()}")
    assert not torch.isnan(features_zero).any(), "Zero gradient should not produce NaN features"
    print("✅ Zero gradient test passed")
    
    print("\n--- Test 2: Very Large Gradient ---")
    large_grad = torch.randn(1000).to(device) * 1000.0
    features_large = server._compute_gradient_features(large_grad, root_grad)
    print(f"Large gradient features: {features_large.cpu().numpy()}")
    assert not torch.isnan(features_large).any(), "Large gradient should not produce NaN features"
    assert not torch.isinf(features_large).any(), "Large gradient should not produce Inf features"
    print("✅ Large gradient test passed")
    
    print("\n--- Test 3: Feature Range Check ---")
    # All features should be in [0, 1] range
    test_grads = [
        torch.randn(1000).to(device) * 0.1,
        torch.randn(1000).to(device) * 2.0,
        -root_grad,
        torch.zeros(1000).to(device)
    ]
    
    for i, grad in enumerate(test_grads):
        features = server._compute_gradient_features(grad, root_grad)
        assert torch.all(features >= 0.0), f"Features should be >= 0, got {features.min().item()}"
        assert torch.all(features <= 1.0), f"Features should be <= 1, got {features.max().item()}"
    
    print("✅ Feature range test passed")

def test_dual_attention_weighting_logic():
    """Test the dual attention weighting logic in detail."""
    print("\n=== Testing Dual Attention Weighting Logic ===")
    
    server = Server()
    device = server.device
    
    # Create controlled feature vectors
    features = torch.tensor([
        [0.1, 0.9, 0.8, 0.2, 0.9, 0.8],  # Good client: low recon error, high similarities, low norm, high consistency
        [0.8, 0.2, 0.3, 0.9, 0.2, 0.1],  # Bad client: high recon error, low similarities, high norm, low consistency
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Neutral client
        [0.2, 0.8, 0.7, 0.8, 0.8, 0.7],  # Mixed client: good features but high norm
    ], device=device)
    
    if not ENABLE_SHAPLEY:
        features = features[:, :5]
    
    # Create dual attention model
    dual_attention = DualAttention(
        feature_dim=features.shape[1],
        hidden_dim=64,
        num_heads=4
    ).to(device)
    
    # Train with synthetic labels
    optimizer = torch.optim.Adam(dual_attention.parameters(), lr=1e-3)
    labels = torch.tensor([1.0, 0.0, 0.5, 0.3], device=device)  # Good, bad, neutral, mixed
    
    for epoch in range(50):
        optimizer.zero_grad()
        trust_scores, _ = dual_attention(features)
        loss = F.mse_loss(trust_scores.squeeze(), labels)
        loss.backward()
        optimizer.step()
    
    # Test final weights
    dual_attention.eval()
    with torch.no_grad():
        trust_scores, confidence = dual_attention(features)
        weights, malicious_indices = dual_attention.get_gradient_weights(features, trust_scores)
    
    print("\nDual Attention Weighting Results:")
    client_types = ["Good", "Bad", "Neutral", "Mixed"]
    for i, client_type in enumerate(client_types):
        print(f"{client_type:<8}: Trust={trust_scores[i].item():.4f}, Weight={weights[i].item():.4f}")
    
    # Verify weighting logic
    assert weights[0] > weights[1], "Good client should have higher weight than bad client"
    assert weights[0] > weights[2], "Good client should have higher weight than neutral client"
    assert 1 in malicious_indices, "Bad client should be detected as malicious"
    
    print("✅ Dual attention weighting logic test passed")

def main():
    """Run all comprehensive tests."""
    print("=== Comprehensive Dual Attention Feature Validation ===")
    
    try:
        # Test 1: Individual feature calculations
        test_individual_features()
        
        # Test 2: Attack detection accuracy
        accuracy = test_attack_detection_accuracy()
        
        # Test 3: Feature robustness
        test_feature_robustness()
        
        # Test 4: Dual attention weighting logic
        test_dual_attention_weighting_logic()
        
        print("\n🎉 ALL COMPREHENSIVE TESTS PASSED! 🎉")
        print("✅ Individual feature calculations are correct")
        print(f"✅ Attack detection accuracy: {accuracy:.1%}")
        print("✅ Feature calculations are robust to edge cases")
        print("✅ Dual attention weighting logic works correctly")
        print("✅ Your dual attention system is ready for production!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 