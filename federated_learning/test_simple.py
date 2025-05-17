import torch
from federated_learning.models.attention import DualAttention

print("Starting test_simple.py...")

def test_dual_attention():
    """Simple test for DualAttention with Shapley values"""
    print("Testing DualAttention with Shapley values")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with 6 features (including Shapley)
    model = DualAttention(
        feature_dim=6,
        hidden_dim=256,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    # Create sample features and add Shapley values
    num_clients = 5
    features = torch.rand((num_clients, 5), device=device)
    shapley = torch.rand((num_clients, 1), device=device)
    enhanced_features = torch.cat([features, shapley], dim=1)
    
    print(f"Features shape: {features.shape}")
    print(f"Enhanced features shape: {enhanced_features.shape}")
    
    # Create global context
    global_context = enhanced_features.mean(dim=0, keepdim=True)
    print(f"Global context shape: {global_context.shape}")
    
    # Test forward pass
    try:
        trust_scores, confidence_scores = model(enhanced_features, global_context)
        print("Forward pass successful!")
        print(f"Trust scores: {trust_scores}")
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")
        return False
    
    # Test gradient weights
    try:
        weights = model.get_gradient_weights(enhanced_features, global_context)
        print("Gradient weights calculation successful!")
        print(f"Weights: {weights}")
        print(f"Weights sum: {weights.sum().item()}")
        return True
    except Exception as e:
        print(f"Gradient weights calculation failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running test_dual_attention()...")
    success = test_dual_attention()
    print(f"Test {'succeeded' if success else 'failed'}")
    print("End of test_simple.py") 