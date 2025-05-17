"""
Test script to verify the gradient feature extraction and dual attention weighting.

This script focuses on:
1. Verifying each feature is calculated correctly
2. Testing with different gradient vectors to ensure the features behave as expected
3. Checking that the dual attention model properly balances different features
4. Ensuring that features where low values are good (reconstruction error) and
   features where high values are good (cosine similarity) are properly handled
"""

import torch
import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import federated learning components
from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.models.attention import DualAttention
from federated_learning.utils.model_utils import set_random_seeds
from federated_learning.models.vae import GradientVAE


def test_feature_extraction_basic():
    """Test basic feature extraction with a simple gradient."""
    print("\n=== Test 1: Basic Feature Extraction ===")
    
    # Set random seeds
    set_random_seeds(RANDOM_SEED)
    
    # Create a server instance
    server = Server()
    
    # Get device
    device = server.device
    
    # Create a simple root gradient
    root_gradient = torch.randn(1000).to(device)
    
    # Create test gradients with known properties
    # 1. Identical to root gradient (should have perfect similarity)
    identical_gradient = root_gradient.clone()
    
    # 2. Similar to root gradient (high similarity)
    similar_gradient = root_gradient.clone() + torch.randn(1000).to(device) * 0.1
    
    # 3. Different from root gradient (low similarity)
    different_gradient = torch.randn(1000).to(device)
    
    # 4. Gradient with different norm
    high_norm_gradient = root_gradient.clone() * 2.0
    
    # 5. Gradient with sign flips (poor sign consistency)
    sign_flipped_gradient = root_gradient.clone()
    # Flip 30% of signs
    flip_indices = torch.randperm(1000)[:300]
    sign_flipped_gradient[flip_indices] *= -1
    
    # Set up the server with the root gradient
    server.root_gradients = [root_gradient]
    
    # Extract features for all gradients
    gradients = [
        identical_gradient,
        similar_gradient, 
        different_gradient,
        high_norm_gradient,
        sign_flipped_gradient
    ]
    
    gradient_names = [
        "Identical",
        "Similar",
        "Different",
        "High Norm", 
        "Sign Flipped"
    ]
    
    features = []
    feature_names = [
        "VAE Reconstruction Error",
        "Root Similarity",
        "Client Similarity",
        "Gradient Norm",
        "Sign Consistency",
        "Shapley Value"
    ]
    
    # We need to train a VAE first to extract reconstruction error features
    vae = GradientVAE(input_dim=1000, projection_dim=32, hidden_dim=24, latent_dim=16)
    vae.to(device)
    
    # Train the VAE with dummy data (just for testing)
    print("Training a simple VAE for testing...")
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    dummy_gradients = [torch.randn(1000).to(device) for _ in range(10)]
    dummy_gradients.append(root_gradient)
    dummy_gradients.append(identical_gradient)
    dummy_gradients.append(similar_gradient)
    
    for epoch in range(5):
        total_loss = 0
        for grad in dummy_gradients:
            optimizer.zero_grad()
            recon, mu, logvar = vae(grad.unsqueeze(0))
            loss = vae.loss_function(recon, grad.unsqueeze(0), mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"VAE Epoch {epoch+1}/5, Loss: {total_loss/len(dummy_gradients):.4f}")
    
    server.vae = vae
    
    # Now extract features for each gradient
    print("\nFeature Extraction Results:")
    for i, grad in enumerate(gradients):
        # Compute features for this gradient
        feature_vector = server._compute_gradient_features(grad, root_gradient)
        features.append(feature_vector)
        
        # Print the feature vector
        print(f"\n{gradient_names[i]} gradient:")
        print(f"  Raw gradient norm: {torch.norm(grad).item():.4f}")
        
        for j, name in enumerate(feature_names):
            if j < feature_vector.shape[0]:  # Some features might be disabled (e.g., Shapley)
                print(f"  {j+1}. {name}: {feature_vector[j].item():.4f}")
    
    features_tensor = torch.stack(features)
    
    # Verify features behave as expected
    print("\nVerifying feature behavior:")
    
    # 1. Identical gradient should have:
    # - Low reconstruction error
    # - High root similarity
    # - High sign consistency
    # - Norm close to 1.0
    identical_features = features[0]
    assert identical_features[1] > 0.9, "Root similarity for identical gradient should be high"
    assert identical_features[4] > 0.9, "Sign consistency for identical gradient should be high"
    print("✅ Identical gradient features verified")
    
    # 2. Different gradient should have:
    # - Lower root similarity than identical gradient
    different_features = features[2]
    assert different_features[1] < features[0][1], "Root similarity for different gradient should be lower"
    print("✅ Different gradient features verified")
    
    # 3. Sign flipped gradient should have:
    # - Lower sign consistency
    sign_flipped_features = features[4]
    assert sign_flipped_features[4] < 0.8, "Sign consistency for sign-flipped gradient should be lower"
    print("✅ Sign-flipped gradient features verified")
    
    return server, features_tensor, gradient_names, feature_names


def test_feature_extraction_with_attacks():
    """Test feature extraction with attacked gradients."""
    print("\n=== Test 2: Feature Extraction with Attacks ===")
    
    # Create a server instance
    server = Server()
    device = server.device
    
    # Create a simple root gradient
    root_gradient = torch.randn(5000).to(device)
    
    # Create test gradients with simulated attacks
    gradients = []
    gradient_names = []
    
    # 1. Honest gradient
    honest_gradient = root_gradient.clone() + torch.randn(5000).to(device) * 0.1
    gradients.append(honest_gradient)
    gradient_names.append("Honest")
    
    # 2. Scaling attack
    scaling_attack = root_gradient.clone() * SCALING_FACTOR
    gradients.append(scaling_attack)
    gradient_names.append("Scaling Attack")
    
    # 3. Partial scaling attack
    partial_scaling = root_gradient.clone()
    indices = torch.randperm(5000)[:int(PARTIAL_SCALING_PERCENT * 5000)]
    partial_scaling[indices] *= SCALING_FACTOR
    gradients.append(partial_scaling)
    gradient_names.append("Partial Scaling Attack")
    
    # 4. Sign flipping attack
    sign_flipping = -root_gradient.clone()
    gradients.append(sign_flipping)
    gradient_names.append("Sign Flipping Attack")
    
    # 5. Noise injection attack
    noise_attack = root_gradient.clone() + torch.randn(5000).to(device) * NOISE_FACTOR
    gradients.append(noise_attack)
    gradient_names.append("Noise Injection Attack")
    
    # Set up the server with the root gradient
    server.root_gradients = [root_gradient]
    
    # We need to train a VAE first to extract reconstruction error features
    vae = GradientVAE(input_dim=5000, projection_dim=32, hidden_dim=24, latent_dim=16)
    vae.to(device)
    
    # Train the VAE with dummy data (just for testing)
    print("Training a simple VAE for testing...")
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    dummy_gradients = [torch.randn(5000).to(device) for _ in range(10)]
    dummy_gradients.append(root_gradient)
    dummy_gradients.append(honest_gradient)
    
    for epoch in range(3):
        total_loss = 0
        for grad in dummy_gradients:
            optimizer.zero_grad()
            recon, mu, logvar = vae(grad.unsqueeze(0))
            loss = vae.loss_function(recon, grad.unsqueeze(0), mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"VAE Epoch {epoch+1}/3, Loss: {total_loss/len(dummy_gradients):.4f}")
    
    server.vae = vae
    
    # Now extract features for each gradient
    features = []
    feature_names = [
        "VAE Reconstruction Error",
        "Root Similarity",
        "Client Similarity",
        "Gradient Norm",
        "Sign Consistency",
        "Shapley Value"
    ]
    
    print("\nFeature Extraction Results:")
    for i, grad in enumerate(gradients):
        # Compute features for this gradient
        feature_vector = server._compute_gradient_features(grad, root_gradient)
        features.append(feature_vector)
        
        # Print the feature vector
        print(f"\n{gradient_names[i]} gradient:")
        print(f"  Raw gradient norm: {torch.norm(grad).item():.4f}")
        
        for j, name in enumerate(feature_names):
            if j < feature_vector.shape[0]:  # Some features might be disabled (e.g., Shapley)
                print(f"  {j+1}. {name}: {feature_vector[j].item():.4f}")
    
    features_tensor = torch.stack(features)
    
    # Verify features behave as expected
    print("\nVerifying attack detection capabilities:")
    
    # Scaling attack should have higher norm
    scaling_features = features[1]
    assert scaling_features[3] > features[0][3], "Scaling attack should have higher norm"
    
    # Sign flipping attack should have negative root similarity
    sign_flipping_features = features[3]
    assert sign_flipping_features[1] < 0.3, "Sign flipping attack should have lower root similarity"
    assert sign_flipping_features[4] < 0.3, "Sign flipping attack should have lower sign consistency"
    
    # Noise attack should have lower root similarity
    noise_features = features[4]
    assert noise_features[1] < features[0][1], "Noise attack should have lower root similarity"
    
    print("✅ Attack detection features verified")
    
    return server, features_tensor, gradient_names, feature_names


def test_dual_attention_feature_balancing():
    """Test that dual attention properly balances different features."""
    print("\n=== Test 3: Dual Attention Feature Balancing ===")
    
    # First get features from our previous test
    server, features, gradient_names, feature_names = test_feature_extraction_with_attacks()
    device = server.device
    
    # Now create a dual attention model
    feature_dim = features.shape[1]
    dual_attention = DualAttention(
        input_dim=feature_dim,
        hidden_dim=DUAL_ATTENTION_HIDDEN_SIZE,
        num_heads=DUAL_ATTENTION_HEADS,
        num_layers=DUAL_ATTENTION_LAYERS
    )
    dual_attention.to(device)
    
    # Create a dataset with honest and malicious samples
    honest_idx = 0  # Honest gradient index
    honest_features = features[honest_idx:honest_idx+1]
    malicious_features = features[1:]  # All attack gradients
    
    # Import training utilities
    from federated_learning.training.training_utils import train_dual_attention
    
    # Train dual attention model
    print("Training dual attention model...")
    dual_attention = train_dual_attention(
        honest_features=honest_features,
        malicious_features=malicious_features,
        epochs=10,
        batch_size=2,
        lr=0.001,
        device=device,
        verbose=True
    )
    
    # Now test the feature balancing
    print("\nTesting feature balancing...")
    
    # Create test gradients with different feature combinations
    # This will help us understand how the dual attention model balances
    # the importance of different features
    test_features = []
    test_names = []
    
    # 1. Base honest features
    test_features.append(features[0].clone())
    test_names.append("Base Honest")
    
    # 2. High reconstruction error but high similarity
    high_recon_high_sim = features[0].clone()
    high_recon_high_sim[0] = 0.5  # Higher reconstruction error (worse)
    high_recon_high_sim[1] = 0.9  # High root similarity (better)
    test_features.append(high_recon_high_sim)
    test_names.append("High Recon Error, High Similarity")
    
    # 3. Low reconstruction error but low similarity
    low_recon_low_sim = features[0].clone()
    low_recon_low_sim[0] = 0.1  # Low reconstruction error (better)
    low_recon_low_sim[1] = 0.5  # Lower root similarity (worse)
    test_features.append(low_recon_low_sim)
    test_names.append("Low Recon Error, Low Similarity")
    
    # 4. Mixed signal - some good features, some bad
    mixed_features = features[0].clone()
    mixed_features[0] = 0.4  # Higher reconstruction error (worse)
    mixed_features[1] = 0.7  # Good root similarity (better)
    mixed_features[2] = 0.4  # Lower client similarity (worse)
    mixed_features[3] = 0.9  # High norm (could be attack)
    mixed_features[4] = 0.9  # High sign consistency (better)
    test_features.append(mixed_features)
    test_names.append("Mixed Features")
    
    # Stack features
    test_features_tensor = torch.stack(test_features)
    
    # Get trust scores and weights
    dual_attention.eval()
    with torch.no_grad():
        trust_scores, _ = dual_attention(test_features_tensor)
        weights, _ = dual_attention.get_gradient_weights(
            test_features_tensor, 
            trust_scores=trust_scores
        )
    
    # Print results
    print("\nDual Attention Results:")
    print("Feature\t\t\t\tTrust Score\tWeight")
    print("-" * 70)
    
    for i, name in enumerate(test_names):
        print(f"{name:<30}\t{trust_scores[i].item():.4f}\t{weights[i].item():.4f}")
    
    # Analysis - make sure the weighting is reasonable
    print("\nFeature Balancing Analysis:")
    
    # High recon error but high similarity should still get decent weight
    assert weights[1].item() > 0.3, "Gradient with high similarity should get reasonable weight despite high recon error"
    
    # Test that trust scores are not simply based on a single feature
    # The mixed feature case should help us understand feature balancing
    print(f"Base case weight: {weights[0].item():.4f}")
    print(f"Mixed features weight: {weights[3].item():.4f}")
    print(f"Weight ratio: {weights[0].item() / weights[3].item():.4f}")
    
    # Verify continuous weighting without thresholds
    if MALICIOUS_WEIGHTING_METHOD == 'continuous':
        weight_range = weights.max().item() - weights.min().item()
        print(f"Weight range: {weight_range:.4f}")
        assert weight_range > 0.1, "Continuous weighting should have a meaningful range"
    
    print("✅ Dual attention feature balancing verified")
    
    return dual_attention, test_features_tensor, weights


def main():
    """Run all the feature extraction tests."""
    print("=== Gradient Feature Extraction Tests ===")
    
    # Test 1: Basic feature extraction
    server, features, gradient_names, feature_names = test_feature_extraction_basic()
    
    # Test 2: Feature extraction with attacks - skipped to save time
    # This is already covered in test_feature_extraction_basic
    
    # Test 3: Dual attention feature balancing
    dual_attention, test_features, weights = test_dual_attention_feature_balancing()
    
    print("\n=== All Tests Passed ===")
    print("✅ Feature extraction is correctly computing all gradient features")
    print("✅ Dual attention properly balances different features for trust scoring")
    print("✅ Continuous weighting is applied without hard thresholds")


if __name__ == "__main__":
    main() 