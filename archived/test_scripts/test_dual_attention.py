import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import traceback
from federated_learning.config.config import *
from federated_learning.models.attention import DualAttention

def test_dual_attention():
    """
    Test that the DualAttention model can handle different feature dimensions.
    """
    print("\n=== Testing DualAttention Model ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print config values
    print(f"DUAL_ATTENTION_FEATURE_DIM: {DUAL_ATTENTION_FEATURE_DIM}")
    print(f"DUAL_ATTENTION_HIDDEN_DIM: {DUAL_ATTENTION_HIDDEN_DIM}")
    print(f"DUAL_ATTENTION_NUM_HEADS: {DUAL_ATTENTION_NUM_HEADS}")
    print(f"DUAL_ATTENTION_DROPOUT: {DUAL_ATTENTION_DROPOUT}")
    
    # Create DualAttention model with expected feature dimension
    try:
        expected_feature_dim = DUAL_ATTENTION_FEATURE_DIM
        hidden_dim = DUAL_ATTENTION_HIDDEN_DIM
        print(f"Creating DualAttention model with feature_dim={expected_feature_dim}, hidden_dim={hidden_dim}")
        model = DualAttention(
            feature_dim=expected_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=DUAL_ATTENTION_NUM_HEADS,
            dropout=DUAL_ATTENTION_DROPOUT
        ).to(device)
        print("Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        traceback.print_exc()
        return
    
    # Test with matching feature dimensions
    print("\nTest 1: Matching feature dimensions")
    batch_size = 5
    features = torch.randn(batch_size, expected_feature_dim, device=device)
    global_context = torch.randn(1, expected_feature_dim, device=device)
    
    print(f"Input features shape: {features.shape}")
    print(f"Global context shape: {global_context.shape}")
    
    try:
        trust_scores, confidence_scores = model(features, global_context)
        print(f"Trust scores shape: {trust_scores.shape}")
        print(f"Confidence scores shape: {confidence_scores.shape}")
        print(f"Trust scores: {trust_scores}")
        print(f"Confidence scores: {confidence_scores}")
        print("Test 1 passed!")
    except Exception as e:
        print(f"Test 1 failed: {str(e)}")
        traceback.print_exc()
    
    # Test with different feature dimensions
    print("\nTest 2: Different feature dimensions")
    actual_feature_dim = 10  # Different from expected
    features = torch.randn(batch_size, actual_feature_dim, device=device)
    global_context = torch.randn(1, actual_feature_dim, device=device)
    
    print(f"Input features shape: {features.shape}")
    print(f"Global context shape: {global_context.shape}")
    
    try:
        trust_scores, confidence_scores = model(features, global_context)
        print(f"Trust scores shape: {trust_scores.shape}")
        print(f"Confidence scores shape: {confidence_scores.shape}")
        print(f"Trust scores: {trust_scores}")
        print(f"Confidence scores: {confidence_scores}")
        print("Test 2 passed!")
    except Exception as e:
        print(f"Test 2 failed: {str(e)}")
        traceback.print_exc()
    
    # Test with different batch sizes
    print("\nTest 3: Different batch sizes")
    batch_size = 10
    features = torch.randn(batch_size, actual_feature_dim, device=device)
    global_context = torch.randn(1, actual_feature_dim, device=device)
    
    print(f"Input features shape: {features.shape}")
    print(f"Global context shape: {global_context.shape}")
    
    try:
        trust_scores, confidence_scores = model(features, global_context)
        print(f"Trust scores shape: {trust_scores.shape}")
        print(f"Confidence scores shape: {confidence_scores.shape}")
        print(f"Trust scores: {trust_scores}")
        print(f"Confidence scores: {confidence_scores}")
        print("Test 3 passed!")
    except Exception as e:
        print(f"Test 3 failed: {str(e)}")
        traceback.print_exc()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_dual_attention() 