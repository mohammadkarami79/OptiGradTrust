"""
Simple Feature Testing - Focus on individual feature validation
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.models.vae import GradientVAE

def test_gradient_norm_feature_detailed():
    """Test gradient norm feature in detail."""
    print("=== Testing Gradient Norm Feature in Detail ===")
    
    server = Server()
    device = server.device
    
    # Create root gradients with known norms
    root_gradients = []
    for _ in range(5):
        grad = torch.randn(1000).to(device) * 0.1  # Small normal gradients
        root_gradients.append(grad)
    
    server.root_gradients = root_gradients
    
    # Test different gradient magnitudes
    test_cases = [
        ("Small honest", torch.randn(1000).to(device) * 0.05),
        ("Normal honest", torch.randn(1000).to(device) * 0.1),
        ("Large honest", torch.randn(1000).to(device) * 0.2),
        ("Scaling attack", torch.randn(1000).to(device) * 2.0),  # 20x larger
        ("Large scaling", torch.randn(1000).to(device) * 10.0), # 100x larger
    ]
    
    print("\nGradient Norm Feature Test:")
    print("Name              | Gradient Norm | Norm Feature | Expected")
    print("-" * 65)
    
    for name, grad in test_cases:
        features = server._compute_gradient_features(grad, root_gradients[0])
        grad_norm = torch.norm(grad).item()
        norm_feature = features[3].item()
        
        # Determine expected behavior
        if "attack" in name.lower() or "scaling" in name.lower():
            expected = "HIGH (>0.8)"
        else:
            expected = "LOW-MED (<0.5)"
        
        print(f"{name:<16} | {grad_norm:>11.4f} | {norm_feature:>10.4f} | {expected}")
    
    # Verify ordering
    features_small = server._compute_gradient_features(test_cases[0][1], root_gradients[0])
    features_attack = server._compute_gradient_features(test_cases[3][1], root_gradients[0])
    features_large_attack = server._compute_gradient_features(test_cases[4][1], root_gradients[0])
    
    assert features_attack[3] > features_small[3], "Attack should have higher norm feature than small gradient"
    assert features_large_attack[3] > features_attack[3], "Larger attack should have higher norm feature"
    
    print("\n✅ Gradient norm feature ordering test passed")

def test_vae_reconstruction_detailed():
    """Test VAE reconstruction feature in detail."""
    print("\n=== Testing VAE Reconstruction Feature in Detail ===")
    
    server = Server()
    device = server.device
    
    # Create and train VAE on specific gradients
    root_grad = torch.randn(1000).to(device) * 0.1
    server.root_gradients = [root_grad]
    
    vae = GradientVAE(input_dim=1000, hidden_dim=64, latent_dim=32).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    # Train VAE on honest-like gradients
    print("Training VAE on honest gradients...")
    for epoch in range(10):
        total_loss = 0
        for _ in range(20):  # More training samples
            honest_grad = torch.randn(1000).to(device) * 0.1
            optimizer.zero_grad()
            recon, mu, logvar = vae(honest_grad.unsqueeze(0))
            loss = vae.loss_function(recon, honest_grad.unsqueeze(0), mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}: Average loss = {total_loss/20:.4f}")
    
    server.vae = vae
    
    # Test reconstruction on different gradient types
    test_cases = [
        ("Similar to training", torch.randn(1000).to(device) * 0.1),
        ("Slightly different", torch.randn(1000).to(device) * 0.15),
        ("Much larger", torch.randn(1000).to(device) * 1.0),
        ("Very different", torch.randn(1000).to(device) * 5.0),
        ("Opposite pattern", -root_grad),
    ]
    
    print("\nVAE Reconstruction Test:")
    print("Name              | Recon Error | Expected")
    print("-" * 45)
    
    for name, grad in test_cases:
        features = server._compute_gradient_features(grad, root_grad)
        recon_error = features[0].item()
        
        if "similar" in name.lower():
            expected = "LOW (<0.3)"
        elif "different" in name.lower() or "opposite" in name.lower():
            expected = "HIGH (>0.7)"
        else:
            expected = "MEDIUM"
        
        print(f"{name:<16} | {recon_error:>9.4f} | {expected}")
    
    # Verify ordering
    similar_features = server._compute_gradient_features(test_cases[0][1], root_grad)
    different_features = server._compute_gradient_features(test_cases[3][1], root_grad)
    
    assert different_features[0] > similar_features[0], "Different gradient should have higher reconstruction error"
    
    print("\n✅ VAE reconstruction test passed")

def test_similarity_features():
    """Test similarity features (root and client)."""
    print("\n=== Testing Similarity Features ===")
    
    server = Server()
    device = server.device
    
    # Create known root gradient
    root_grad = torch.randn(1000).to(device)
    root_grad = root_grad / torch.norm(root_grad)  # Normalize for predictable similarity
    server.root_gradients = [root_grad]
    
    test_cases = [
        ("Identical", root_grad.clone()),
        ("Scaled identical", root_grad * 2.0),
        ("Opposite", -root_grad),
        ("Orthogonal", torch.randn(1000).to(device)),
        ("Random", torch.randn(1000).to(device) * 0.5),
    ]
    
    # Make orthogonal gradient truly orthogonal
    orthogonal_grad = test_cases[3][1]
    orthogonal_grad = orthogonal_grad - torch.dot(orthogonal_grad, root_grad) * root_grad
    orthogonal_grad = orthogonal_grad / torch.norm(orthogonal_grad)
    test_cases[3] = ("Orthogonal", orthogonal_grad)
    
    print("\nSimilarity Features Test:")
    print("Name              | Root Sim | Expected  | Actual Cosine")
    print("-" * 55)
    
    for name, grad in test_cases:
        features = server._compute_gradient_features(grad, root_grad)
        root_similarity = features[1].item()
        
        # Calculate actual cosine similarity for verification
        actual_cosine = torch.cosine_similarity(grad.unsqueeze(0), root_grad.unsqueeze(0)).item()
        normalized_cosine = (actual_cosine + 1) / 2  # Convert from [-1,1] to [0,1]
        
        if name == "Identical":
            expected = "~1.0"
        elif name == "Scaled identical":
            expected = "~1.0"
        elif name == "Opposite":
            expected = "~0.0"
        elif name == "Orthogonal":
            expected = "~0.5"
        else:
            expected = "varies"
        
        print(f"{name:<16} | {root_similarity:>6.3f}   | {expected:<8} | {normalized_cosine:>6.3f}")
    
    # Verify key relationships
    identical_sim = server._compute_gradient_features(test_cases[0][1], root_grad)[1]
    opposite_sim = server._compute_gradient_features(test_cases[2][1], root_grad)[1]
    orthogonal_sim = server._compute_gradient_features(test_cases[3][1], root_grad)[1]
    
    assert identical_sim > 0.95, f"Identical should have similarity ~1.0, got {identical_sim:.3f}"
    assert opposite_sim < 0.05, f"Opposite should have similarity ~0.0, got {opposite_sim:.3f}"
    assert 0.4 < orthogonal_sim < 0.6, f"Orthogonal should have similarity ~0.5, got {orthogonal_sim:.3f}"
    
    print("\n✅ Similarity features test passed")

def test_sign_consistency():
    """Test sign consistency feature."""
    print("\n=== Testing Sign Consistency Feature ===")
    
    server = Server()
    device = server.device
    
    root_grad = torch.randn(1000).to(device)
    server.root_gradients = [root_grad]
    
    test_cases = [
        ("Same signs", torch.abs(root_grad) * torch.sign(root_grad)),
        ("Opposite signs", -torch.abs(root_grad) * torch.sign(root_grad)),
        ("50% flipped", torch.where(torch.randperm(1000).to(device) < 500, root_grad, -root_grad)),
        ("Random signs", torch.randn(1000).to(device)),
    ]
    
    print("\nSign Consistency Test:")
    print("Name              | Sign Consistency | Expected")
    print("-" * 50)
    
    for name, grad in test_cases:
        features = server._compute_gradient_features(grad, root_grad)
        sign_consistency = features[4].item()
        
        if "same" in name.lower():
            expected = "~1.0"
        elif "opposite" in name.lower():
            expected = "~0.0"
        elif "50%" in name.lower():
            expected = "~0.5"
        else:
            expected = "varies"
        
        print(f"{name:<16} | {sign_consistency:>14.3f} | {expected}")
    
    # Verify relationships
    same_consistency = server._compute_gradient_features(test_cases[0][1], root_grad)[4]
    opposite_consistency = server._compute_gradient_features(test_cases[1][1], root_grad)[4]
    
    assert same_consistency > 0.95, f"Same signs should have consistency ~1.0, got {same_consistency:.3f}"
    assert opposite_consistency < 0.05, f"Opposite signs should have consistency ~0.0, got {opposite_consistency:.3f}"
    
    print("\n✅ Sign consistency test passed")

def main():
    """Run all focused feature tests."""
    print("=== Focused Feature Validation Tests ===")
    
    try:
        test_gradient_norm_feature_detailed()
        test_vae_reconstruction_detailed()
        test_similarity_features()
        test_sign_consistency()
        
        print("\n🎉 ALL FOCUSED TESTS PASSED! 🎉")
        print("✅ Gradient norm feature correctly distinguishes attack gradients")
        print("✅ VAE reconstruction distinguishes between similar and different gradients")
        print("✅ Similarity features work as expected")
        print("✅ Sign consistency feature works correctly")
        print("✅ All individual features are working properly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 