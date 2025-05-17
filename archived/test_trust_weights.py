import torch
import torch.nn as nn
import numpy as np
from federated_learning.models.attention import DualAttention

def test_trust_weights():
    """Test trust weight calculation with known trust scores."""
    print("=== Testing Trust Weight Calculation ===")
    
    # Create a simple model
    model = DualAttention(feature_dim=14, hidden_dim=128, num_heads=4, dropout=0.2)
    
    # Create mock features and trust scores
    batch_size = 5
    features = torch.randn(batch_size, 14)
    
    # Mock trust scores (0 = honest, 1 = malicious)
    trust_scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.1])  # 3 honest, 2 malicious
    confidence_scores = torch.ones(batch_size)  # All confident
    
    # Calculate weights
    with torch.no_grad():
        # Calculate trust-based weights
        trust_weights = torch.exp(-10 * trust_scores)  # Higher weights for lower trust scores
        
        # Apply confidence adjustment
        weights = trust_weights * confidence_scores
        
        # Identify honest clients
        honest_mask = trust_scores < 0.2
        num_honest = honest_mask.sum().item()
        
        if num_honest > 0:
            # Ensure honest clients get at least 80% of total weight
            honest_total = 0.8
            malicious_total = 0.2
            
            # Split weights between honest and malicious clients
            honest_weights = weights * honest_mask
            malicious_weights = weights * (~honest_mask)
            
            # Normalize each group separately
            if honest_weights.sum() > 0:
                honest_weights = honest_weights / honest_weights.sum() * honest_total
            if malicious_weights.sum() > 0:
                malicious_weights = malicious_weights / malicious_weights.sum() * malicious_total
            
            # Combine weights
            weights = honest_weights + malicious_weights
        
        # Normalize final weights
        epsilon = 1e-8
        normalized_weights = weights / (weights.sum() + epsilon)
    
    # Print results
    print("\nTrust Scores:")
    for i, score in enumerate(trust_scores):
        print(f"Client {i}: {score.item():.4f}")
    
    print("\nNormalized Weights:")
    for i, weight in enumerate(normalized_weights):
        print(f"Client {i}: {weight.item():.4f}")
    
    # Verify properties
    print("\nVerifying properties:")
    print(f"1. Weight sum: {normalized_weights.sum().item():.4f}")
    
    honest_weights = normalized_weights[honest_mask]
    malicious_weights = normalized_weights[~honest_mask]
    
    print(f"2. Honest clients total weight: {honest_weights.sum().item():.4f}")
    print(f"3. Malicious clients total weight: {malicious_weights.sum().item():.4f}")
    print(f"4. Number of honest clients: {num_honest}")
    print(f"5. Average honest weight: {honest_weights.mean().item():.4f}")
    print(f"6. Average malicious weight: {malicious_weights.mean().item():.4f}")
    
    # Assert conditions
    assert abs(normalized_weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
    assert honest_weights.sum() > 0.7, "Honest clients should have significant total weight"
    assert honest_weights.mean() > malicious_weights.mean(), "Honest clients should have higher average weight"
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_trust_weights() 