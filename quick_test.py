#!/usr/bin/env python3
"""
Quick test to verify trust score fixes are working.
"""

import torch
import sys
import os

# Add federated_learning to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'federated_learning'))

def quick_test():
    """Quick test of trust score logic."""
    print("=== Quick Trust Score Test ===")
    
    # Create mock dual attention model
    from federated_learning.models.attention import DualAttention
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dual_attention = DualAttention(feature_dim=5, hidden_dim=64).to(device)
    
    # Create test features - clear honest vs malicious differences
    honest_features = torch.tensor([
        [0.1, 0.9, 0.8, 0.2, 0.9],  # Very honest features: low recon error, high similarities, low norm, high consistency
        [0.2, 0.8, 0.7, 0.3, 0.8],  # Honest features
    ], device=device, dtype=torch.float32)
    
    malicious_features = torch.tensor([
        [0.9, 0.1, 0.2, 0.9, 0.1],  # Very malicious features: high recon error, low similarities, high norm, low consistency  
        [0.8, 0.2, 0.3, 0.8, 0.2],  # Malicious features
    ], device=device, dtype=torch.float32)
    
    all_features = torch.cat([honest_features, malicious_features], dim=0)
    
    print("Input features:")
    print("Honest clients:")
    for i in range(honest_features.shape[0]):
        print(f"  Client {i}: {honest_features[i].cpu().numpy()}")
    print("Malicious clients:")
    for i in range(malicious_features.shape[0]):
        print(f"  Client {i+2}: {malicious_features[i].cpu().numpy()}")
    
    # Test dual attention output
    with torch.no_grad():
        malicious_scores, confidence = dual_attention(all_features)
    
    # Convert to trust scores (1 - malicious_score)
    trust_scores = 1.0 - malicious_scores
    
    print("\nModel outputs:")
    print("Malicious scores:", malicious_scores.detach().cpu().numpy())
    print("Trust scores (1 - malicious):", trust_scores.detach().cpu().numpy())
    
    # Check the logic
    honest_trust_avg = trust_scores[:2].mean().item()
    malicious_trust_avg = trust_scores[2:].mean().item()
    
    print(f"\nAverage scores:")
    print(f"Honest clients - Trust: {honest_trust_avg:.4f}, Malicious: {malicious_scores[:2].mean().item():.4f}")
    print(f"Malicious clients - Trust: {malicious_trust_avg:.4f}, Malicious: {malicious_scores[2:].mean().item():.4f}")
    
    # Test weight computation (as it would be done in the server)
    weights = [max(0.01, score.item()) for score in trust_scores]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    print(f"\nAggregation weights (based on trust scores):")
    for i, w in enumerate(weights):
        client_type = "Honest" if i < 2 else "Malicious"
        print(f"  Client {i} ({client_type}): {w:.4f}")
    
    # Verify correctness
    honest_weight_avg = (weights[0] + weights[1]) / 2
    malicious_weight_avg = (weights[2] + weights[3]) / 2
    
    print(f"\nWeight averages:")
    print(f"Honest clients: {honest_weight_avg:.4f}")
    print(f"Malicious clients: {malicious_weight_avg:.4f}")
    
    if honest_trust_avg > malicious_trust_avg:
        print("✅ PASS: Honest clients have higher trust scores")
        if honest_weight_avg > malicious_weight_avg:
            print("✅ PASS: Honest clients have higher aggregation weights")
            print("✅ OVERALL: Trust score logic is working correctly!")
            return True
        else:
            print("❌ FAIL: Weight computation is incorrect")
            return False
    else:
        print("❌ FAIL: Trust score logic is incorrect")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Quick test PASSED! The trust score fixes are working.")
    else:
        print("\n💥 Quick test FAILED! Trust score logic needs more fixes.")
    sys.exit(0 if success else 1) 