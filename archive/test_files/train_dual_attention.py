import torch
import numpy as np
import random
import os
from federated_learning.models.attention import DualAttention
from federated_learning.training.training_utils import train_dual_attention
from federated_learning.config.config import *

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def main():
    """Train a dual attention model on synthetic data"""
    print("\n=== Training Dual Attention Model on Synthetic Data ===")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directory for model weights if it doesn't exist
    os.makedirs("model_weights", exist_ok=True)
    
    # Set up parameters
    feature_dim = 6 if ENABLE_SHAPLEY else 5
    num_honest = 100  # Generate more samples for better training
    num_malicious = 100
    
    print(f"Generating {num_honest} honest and {num_malicious} malicious samples with {feature_dim} features")
    
    # Create honest features with good characteristics
    honest_features = []
    
    # Create a range of honest features with some variance
    for i in range(num_honest):
        # Start with base values
        feature = torch.zeros(feature_dim, device=device)
        
        # Set features for honest clients (with some noise)
        # Lower reconstruction error is better (0 is perfect)
        feature[0] = 0.2 + 0.2 * torch.rand(1, device=device).item()  # Reconstruction error (0.2-0.4)
        feature[1] = 0.7 + 0.2 * torch.rand(1, device=device).item()  # Root similarity (0.7-0.9)
        feature[2] = 0.7 + 0.2 * torch.rand(1, device=device).item()  # Client similarity (0.7-0.9)
        feature[3] = 0.4 + 0.2 * torch.rand(1, device=device).item()  # Gradient norm (0.4-0.6)
        feature[4] = 0.7 + 0.2 * torch.rand(1, device=device).item()  # Consistency (0.7-0.9)
        
        if feature_dim > 5:
            feature[5] = 0.7 + 0.2 * torch.rand(1, device=device).item()  # Shapley value (0.7-0.9)
        
        honest_features.append(feature)
    
    honest_features = torch.stack(honest_features)
    
    # Create different types of malicious features
    malicious_features = []
    
    # Generate various attack patterns
    attack_types = [
        "scaling_attack",       # High norm
        "sign_flipping_attack", # Low similarity
        "noise_attack",         # High reconstruction error
        "combined_attack",      # Multiple issues
        "partial_attack",       # Subtle issues
        "adaptive_attack"       # Tries to evade detection
    ]
    
    attacks_per_type = num_malicious // len(attack_types)
    
    # Create malicious features for each attack type
    for attack_type in attack_types:
        for i in range(attacks_per_type):
            feature = torch.zeros(feature_dim, device=device)
            
            # Base values - start with somewhat normal values
            # Then customize based on attack type
            feature[0] = 0.3 + 0.1 * torch.rand(1, device=device).item()  # Reconstruction
            feature[1] = 0.6 + 0.1 * torch.rand(1, device=device).item()  # Root similarity
            feature[2] = 0.6 + 0.1 * torch.rand(1, device=device).item()  # Client similarity
            feature[3] = 0.5 + 0.1 * torch.rand(1, device=device).item()  # Norm
            feature[4] = 0.6 + 0.1 * torch.rand(1, device=device).item()  # Consistency
            
            if feature_dim > 5:
                feature[5] = 0.5 + 0.1 * torch.rand(1, device=device).item()  # Shapley value
            
            # Now modify based on attack type
            if attack_type == "scaling_attack":
                # Higher norm
                feature[3] = 0.8 + 0.15 * torch.rand(1, device=device).item()
                feature[0] = 0.4 + 0.2 * torch.rand(1, device=device).item()
                
            elif attack_type == "sign_flipping_attack":
                # Lower similarity to root and other clients
                feature[1] = 0.2 + 0.2 * torch.rand(1, device=device).item()
                feature[2] = 0.2 + 0.2 * torch.rand(1, device=device).item()
                
            elif attack_type == "noise_attack":
                # Higher reconstruction error, lower consistency
                feature[0] = 0.7 + 0.2 * torch.rand(1, device=device).item()
                feature[4] = 0.3 + 0.2 * torch.rand(1, device=device).item()
                
            elif attack_type == "combined_attack":
                # Multiple issues
                feature[0] = 0.6 + 0.2 * torch.rand(1, device=device).item()  # Higher recon error
                feature[1] = 0.3 + 0.2 * torch.rand(1, device=device).item()  # Lower root similarity
                feature[3] = 0.7 + 0.2 * torch.rand(1, device=device).item()  # Higher norm
                
            elif attack_type == "partial_attack":
                # Subtle issues, harder to detect
                feature[0] = 0.4 + 0.1 * torch.rand(1, device=device).item()
                feature[1] = 0.5 + 0.1 * torch.rand(1, device=device).item()
                feature[3] = 0.6 + 0.1 * torch.rand(1, device=device).item()
                
            elif attack_type == "adaptive_attack":
                # Tries to maintain good metrics while being malicious
                feature[0] = 0.3 + 0.1 * torch.rand(1, device=device).item()  # Good reconstruction
                feature[1] = 0.6 + 0.1 * torch.rand(1, device=device).item()  # Decent similarity
                feature[3] = 0.5 + 0.1 * torch.rand(1, device=device).item()  # Normal norm
                feature[4] = 0.4 + 0.1 * torch.rand(1, device=device).item()  # Lower consistency
                
            if feature_dim > 5:
                if attack_type in ["scaling_attack", "sign_flipping_attack", "combined_attack"]:
                    feature[5] = 0.2 + 0.2 * torch.rand(1, device=device).item()  # Low Shapley
                else:
                    feature[5] = 0.4 + 0.1 * torch.rand(1, device=device).item()  # Medium Shapley
            
            malicious_features.append(feature)
    
    # Fill in any remaining slots if needed
    remaining = num_malicious - len(malicious_features)
    for i in range(remaining):
        # Random attack type
        feature = torch.zeros(feature_dim, device=device)
        
        # Bad values for all metrics
        feature[0] = 0.5 + 0.3 * torch.rand(1, device=device).item()  # High reconstruction error
        feature[1] = 0.3 + 0.3 * torch.rand(1, device=device).item()  # Low similarity
        feature[2] = 0.3 + 0.3 * torch.rand(1, device=device).item()  # Low similarity
        feature[3] = 0.7 + 0.3 * torch.rand(1, device=device).item()  # High norm
        feature[4] = 0.3 + 0.3 * torch.rand(1, device=device).item()  # Low consistency
        
        if feature_dim > 5:
            feature[5] = 0.2 + 0.2 * torch.rand(1, device=device).item()  # Low Shapley value
        
        malicious_features.append(feature)
    
    # Stack all malicious features
    malicious_features = torch.stack(malicious_features)
    
    # Create labels (0 for honest, 1 for malicious) - updated convention
    honest_labels = torch.zeros(num_honest, device=device)
    malicious_labels = torch.ones(num_malicious, device=device)
    
    # Combine features and labels
    all_features = torch.cat([honest_features, malicious_features], dim=0)
    all_labels = torch.cat([honest_labels, malicious_labels], dim=0)
    
    # Print statistics
    print(f"Generated {len(honest_features)} honest features and {len(malicious_features)} malicious features")
    print(f"Combined shape: {all_features.shape}")
    
    # Analyze feature distributions
    print("\nFeature statistics:")
    for i in range(feature_dim):
        honest_mean = honest_features[:, i].mean().item()
        malicious_mean = malicious_features[:, i].mean().item()
        print(f"Feature {i}: Honest mean = {honest_mean:.3f}, Malicious mean = {malicious_mean:.3f}, Difference = {honest_mean - malicious_mean:.3f}")
    
    # Train dual attention model
    print("\nTraining dual attention model...")
    model = train_dual_attention(
        honest_features=honest_features,
        malicious_features=malicious_features,
        epochs=100,
        batch_size=32,
        lr=0.001
    )
    
    # Test on original data
    print("\nEvaluating trained model...")
    model.eval()
    with torch.no_grad():
        malicious_scores, confidence = model(all_features)
        predictions = (malicious_scores >= 0.5).float()
        
        # Calculate accuracy
        accuracy = (predictions == all_labels).float().mean().item()
        
        # Calculate honest and malicious accuracy
        honest_acc = (predictions[:num_honest] == honest_labels).float().mean().item()
        malicious_acc = (predictions[num_honest:] == malicious_labels).float().mean().item()
        
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Honest accuracy: {honest_acc:.4f}")
        print(f"Malicious accuracy: {malicious_acc:.4f}")
        
        # Print a few examples
        print("\nSample predictions:")
        for i in range(5):
            idx = i
            true_label = "Honest" if all_labels[idx].item() < 0.5 else "Malicious"
            pred_label = "Honest" if predictions[idx].item() < 0.5 else "Malicious"
            print(f"Sample {i} (Honest): True={true_label}, Pred={pred_label}, Score={malicious_scores[idx].item():.4f}")
        
        for i in range(5):
            idx = num_honest + i
            true_label = "Honest" if all_labels[idx].item() < 0.5 else "Malicious"
            pred_label = "Honest" if predictions[idx].item() < 0.5 else "Malicious"
            print(f"Sample {i} (Malicious): True={true_label}, Pred={pred_label}, Score={malicious_scores[idx].item():.4f}")
    
    # Final check: run dual attention get_gradient_weights to see weight distribution
    weights, detected_malicious = model.get_gradient_weights(all_features)
    
    # Check weight differentiation
    honest_weights = weights[:num_honest]
    malicious_weights = weights[num_honest:]
    
    # Higher weight ratio means honest clients get more weight (better)
    # Since weights are now (1 - malicious_score), honest clients should have higher weights
    weight_ratio = honest_weights.mean() / malicious_weights.mean()
    
    print(f"\nWeight distribution analysis:")
    print(f"Honest clients avg weight: {honest_weights.mean().item():.4f}")
    print(f"Malicious clients avg weight: {malicious_weights.mean().item():.4f}")
    print(f"Weight ratio (honest/malicious): {weight_ratio:.2f}x")
    print(f"Detected {len(detected_malicious)} malicious clients")
    
    if weight_ratio > 1.5:
        print("\n✅ SUCCESS: Dual attention model successfully trained!")
        print(f"Honest clients receive {weight_ratio:.2f}x more weight than malicious clients")
    else:
        print("\n⚠️ WARNING: Dual attention model may need more training.")
        print(f"Weight ratio of {weight_ratio:.2f}x is below the target of 1.5x")

if __name__ == "__main__":
    main() 