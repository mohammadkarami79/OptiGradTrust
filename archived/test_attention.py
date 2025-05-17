import torch
import traceback
from federated_learning.models.attention import DualAttention

def test_dual_attention():
    print("=" * 50)
    print("TESTING DUAL ATTENTION MODEL FIX")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the model
    print("\nInitializing DualAttention model...")
    feature_dim = 4
    model = DualAttention(feature_dim=feature_dim).to(device)
    print(f"Model created with feature_dim={feature_dim}")
    
    # Create sample input with batch size > 1
    batch_size = 10
    features = torch.randn(batch_size, feature_dim, device=device)
    
    print(f"\nTesting with features shape: {features.shape}")
    
    # Test case 1: Using global_context with batch_size=1
    print("\nTest case 1: Global context with batch_size=1")
    global_context = torch.randn(1, feature_dim, device=device)  # Global context has batch_size=1
    print(f"Global context shape: {global_context.shape}")
    
    try:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            print("Processing features with global context...")
            output = model(features, global_context)
            print(f"Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
    # Test case 2: Using global_context with same batch_size
    print("\nTest case 2: Context with same batch_size")
    same_batch_context = torch.randn(batch_size, feature_dim, device=device)
    print(f"Context shape: {same_batch_context.shape}")
    
    try:
        with torch.no_grad():
            print("Processing features with same-batch context...")
            output = model(features, same_batch_context)
            print(f"Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
    # Test case 3: Using global_context with different batch_size > 1
    print("\nTest case 3: Context with different batch_size > 1")
    diff_batch_context = torch.randn(5, feature_dim, device=device)
    print(f"Context shape: {diff_batch_context.shape}")
    
    try:
        with torch.no_grad():
            print("Processing features with different-batch context...")
            output = model(features, diff_batch_context)
            print(f"Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
    # Test case 4: No global context
    print("\nTest case 4: No global context")
    try:
        with torch.no_grad():
            print("Processing features without global context...")
            output = model(features)
            print(f"Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    
    print("\nAll tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_dual_attention() 