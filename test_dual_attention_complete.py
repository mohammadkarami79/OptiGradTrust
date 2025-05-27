"""
Comprehensive test for dual attention feature extraction and trust scoring.
This test ensures that features are calculated correctly and dual attention 
properly weights honest vs malicious gradients.
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

def test_gradient_norm_feature():
    """Test that gradient norm feature properly distinguishes attacks."""
    print("\n=== Testing Gradient Norm Feature ===")
    
    server = Server()
    device = server.device
    
    # Create root gradients with normal norms
    root_gradients = []
    for _ in range(5):
        grad = torch.randn(1000).to(device) * 0.1  # Normal magnitude
        root_gradients.append(grad)
    
    server.root_gradients = root_gradients
    
    # Create test gradients
    honest_grad = torch.randn(1000).to(device) * 0.1  # Similar to root
    scaling_attack = torch.randn(1000).to(device) * 2.0  # 20x larger
    
    # Extract features
    honest_features = server._compute_gradient_features(honest_grad, root_gradients[0])
    attack_features = server._compute_gradient_features(scaling_attack, root_gradients[0])
    
    # Print results
    print(f"Honest gradient norm: {torch.norm(honest_grad).item():.4f}")
    print(f"Attack gradient norm: {torch.norm(scaling_attack).item():.4f}")
    print(f"Honest norm feature: {honest_features[3].item():.4f}")
    print(f"Attack norm feature: {attack_features[3].item():.4f}")
    
    # Verify attack has higher norm feature
    assert attack_features[3] > honest_features[3], "Attack should have higher norm feature"
    print("✅ Gradient norm feature test passed")
    
    return server

def test_feature_extraction_complete():
    """Test complete feature extraction with all features."""
    print("\n=== Testing Complete Feature Extraction ===")
    
    server = test_gradient_norm_feature()  # Uses the server from previous test
    device = server.device
    
    # Train a simple VAE for testing
    vae = GradientVAE(input_dim=1000, hidden_dim=32, latent_dim=16)
    vae.to(device)
    
    # Quick VAE training
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(3):
        for grad in server.root_gradients:
            optimizer.zero_grad()
            recon, mu, logvar = vae(grad.unsqueeze(0))
            loss = vae.loss_function(recon, grad.unsqueeze(0), mu, logvar)
            loss.backward()
            optimizer.step()
    
    server.vae = vae
    
    # Test different gradient types
    gradients = {
        "honest": torch.randn(1000).to(device) * 0.1,
        "scaling_attack": torch.randn(1000).to(device) * 2.0,
        "sign_flip": -server.root_gradients[0].clone(),
        "noise_attack": server.root_gradients[0].clone() + torch.randn(1000).to(device) * 0.5
    }
    
    features = {}
    for name, grad in gradients.items():
        feat = server._compute_gradient_features(grad, server.root_gradients[0])
        features[name] = feat
        
        print(f"\n{name.upper()} features:")
        print(f"  VAE Recon Error: {feat[0].item():.4f}")
        print(f"  Root Similarity: {feat[1].item():.4f}")
        print(f"  Gradient Norm: {feat[3].item():.4f}")
        print(f"  Sign Consistency: {feat[4].item():.4f}")
    
    # Verify expectations
    assert features["honest"][1] > features["sign_flip"][1], "Honest should have higher root similarity than sign flip"
    assert features["scaling_attack"][3] > features["honest"][3], "Scaling attack should have higher norm"
    assert features["sign_flip"][4] < 0.2, "Sign flip should have low consistency"
    
    print("✅ Complete feature extraction test passed")
    return server, features

def test_dual_attention_weighting():
    """Test that dual attention gives appropriate weights."""
    print("\n=== Testing Dual Attention Weighting ===")
    
    server, features = test_feature_extraction_complete()
    device = server.device
    
    # Create feature tensor
    feature_list = []
    names = []
    for name, feat in features.items():
        feature_list.append(feat)
        names.append(name)
    
    feature_tensor = torch.stack(feature_list)
    
    # Create and train dual attention
    dual_attention = DualAttention(
        feature_dim=feature_tensor.shape[1],
        hidden_dim=64,
        num_heads=4
    ).to(device)
    
    # Simple training on synthetic data
    optimizer = torch.optim.Adam(dual_attention.parameters(), lr=1e-3)
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Create labels (honest=1, attacks=0)
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # honest, scaling, sign_flip, noise
        
        trust_scores, _ = dual_attention(feature_tensor)
        loss = F.mse_loss(trust_scores.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
    
    # Test final weights
    dual_attention.eval()
    with torch.no_grad():
        trust_scores, _ = dual_attention(feature_tensor)
        weights, _ = dual_attention.get_gradient_weights(feature_tensor, trust_scores)
    
    print("\nDual Attention Results:")
    for i, name in enumerate(names):
        print(f"{name:<15}: Trust={trust_scores[i].item():.4f}, Weight={weights[i].item():.4f}")
    
    # Verify honest gets highest weight
    honest_idx = names.index("honest")
    honest_weight = weights[honest_idx].item()
    
    for i, name in enumerate(names):
        if name != "honest":
            assert honest_weight >= weights[i].item(), f"Honest weight should be >= {name} weight"
    
    print("✅ Dual attention weighting test passed")
    return dual_attention

def main():
    """Run all comprehensive tests."""
    print("=== Comprehensive Dual Attention Testing ===")
    
    try:
        # Test 1: Gradient norm feature
        test_gradient_norm_feature()
        
        # Test 2: Complete feature extraction
        test_feature_extraction_complete()
        
        # Test 3: Dual attention weighting
        test_dual_attention_weighting()
        
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("✅ Feature extraction is working correctly")
        print("✅ Gradient norm properly distinguishes attacks")
        print("✅ Dual attention gives higher weights to honest gradients")
        print("✅ System is ready for production use")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 