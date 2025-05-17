import torch
import torch.nn as nn
import numpy as np
from federated_learning.models.attention import DualAttention
from federated_learning.training.aggregation import extract_gradient_features

def generate_test_gradients():
    """Generate test gradients with different attack patterns"""
    # Create a base gradient
    base_grad = torch.randn(1000)  # 1000-dimensional gradient
    base_grad = base_grad / torch.norm(base_grad)  # Normalize
    
    gradients = []
    attack_types = []
    
    # 1. Honest clients (3)
    for _ in range(3):
        # Add small random noise to base gradient
        noise = torch.randn_like(base_grad) * 0.1
        grad = base_grad + noise
        grad = grad / torch.norm(grad)  # Normalize
        gradients.append(grad)
        attack_types.append('honest')
    
    # 2. Sign-flip attacks (2)
    for scale in [1.0, 1.5]:
        grad = -base_grad * scale  # Flip sign and scale
        gradients.append(grad)
        attack_types.append('sign_flip')
    
    # 3. Scale attacks (2)
    for scale in [5.0, 10.0]:
        grad = base_grad * scale  # Scale up
        gradients.append(grad)
        attack_types.append('scale')
    
    # 4. Pattern attacks (2)
    for _ in range(2):
        # Create a gradient with similar direction but different pattern
        pattern = torch.randn_like(base_grad)
        pattern = pattern / torch.norm(pattern)
        # Mix with base gradient to maintain some similarity
        grad = 0.7 * base_grad + 0.3 * pattern
        grad = grad / torch.norm(grad)
        gradients.append(grad)
        attack_types.append('pattern')
    
    # 5. Random noise attacks (2)
    for _ in range(2):
        grad = torch.randn_like(base_grad)
        grad = grad / torch.norm(grad)
        gradients.append(grad)
        attack_types.append('random')
    
    return gradients, attack_types, base_grad

def train_model(model, features, attack_types, num_epochs=100):
    """Train the DualAttention model."""
    print("\nTraining DualAttention model...")
    
    # Create labels (0 for honest, 1 for malicious)
    labels = torch.tensor([1.0 if t != 'honest' else 0.0 for t in attack_types], dtype=torch.float32)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        trust_scores, confidence = model(features)
        
        # Calculate loss
        loss = criterion(trust_scores, labels)
        
        # Add confidence regularization
        confidence_target = (trust_scores.detach() > 0.5).float()
        confidence_loss = criterion(confidence, confidence_target)
        
        # Add separation loss to encourage clear distinction
        honest_scores = trust_scores[labels == 0]
        malicious_scores = trust_scores[labels == 1]
        
        if len(honest_scores) > 0 and len(malicious_scores) > 0:
            separation_loss = torch.mean(
                torch.relu(honest_scores.mean() - 0.2) +  # Honest scores should be < 0.2
                torch.relu(0.8 - malicious_scores.mean()) +  # Malicious scores should be > 0.8
                torch.relu(0.6 - (malicious_scores.mean() - honest_scores.mean()))  # Ensure clear separation
            )
        else:
            separation_loss = torch.tensor(0.0)
        
        # Total loss
        total_loss = loss + 0.2 * confidence_loss + 1.0 * separation_loss  # Increased separation loss weight
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")
            if len(honest_scores) > 0 and len(malicious_scores) > 0:
                print(f"  Honest scores: {honest_scores.mean().item():.4f}, Malicious scores: {malicious_scores.mean().item():.4f}")
                print(f"  Separation: {(malicious_scores.mean() - honest_scores.mean()).item():.4f}")
    
    print("Training completed!")

def test_gradient_weighting():
    print("=== Testing Enhanced Gradient Weighting ===")
    
    try:
        print("Generating test gradients...")
        gradients, attack_types, base_grad = generate_test_gradients()
        print(f"Generated {len(gradients)} gradients with types: {set(attack_types)}")
        
        print("\nExtracting features...")
        features = extract_gradient_features(gradients, base_grad)
        print(f"Feature shape: {features.shape}")
        
        print("\nInitializing DualAttention model...")
        model = DualAttention(
            feature_dim=features.shape[1],
            hidden_dim=128,
            num_heads=4,
            dropout=0.2
        )
        print("Model initialized successfully")
        
        # Train the model
        train_model(model, features, attack_types)
        
        print("\nCalculating gradient weights...")
        weights = model.get_gradient_weights(features)
        print(f"Generated weights shape: {weights.shape}")
        
        # Analyze results
        print("\nWeight Analysis by Attack Type:")
        
        # Group weights by attack type
        weight_groups = {}
        for weight, attack_type in zip(weights.detach(), attack_types):
            if attack_type not in weight_groups:
                weight_groups[attack_type] = []
            weight_groups[attack_type].append(weight.item())
        
        # Print statistics for each attack type
        for attack_type, weights_list in weight_groups.items():
            mean_weight = np.mean(weights_list)
            std_weight = np.std(weights_list)
            print(f"\n{attack_type.upper()}:")
            print(f"  Mean weight: {mean_weight:.4f}")
            print(f"  Std weight: {std_weight:.4f}")
            print(f"  Weights: {[f'{w:.4f}' for w in weights_list]}")
        
        # Verify weight properties
        print("\nVerifying weight properties:")
        
        # 1. Check weight normalization
        weight_sum = weights.sum().item()
        print(f"1. Weight sum: {weight_sum:.4f} (should be close to 1.0)")
        assert abs(weight_sum - 1.0) < 1e-6, "Weights should sum to 1"
        
        # 2. Check honest client weights
        honest_weights = [w.item() for w, t in zip(weights.detach(), attack_types) if t == 'honest']
        honest_mean = np.mean(honest_weights)
        print(f"2. Honest client mean weight: {honest_mean:.4f}")
        assert honest_mean > 0.1, "Honest clients should have significant weights"
        
        # 3. Check malicious client weights
        malicious_weights = [w.item() for w, t in zip(weights.detach(), attack_types) if t != 'honest']
        malicious_mean = np.mean(malicious_weights)
        print(f"3. Malicious client mean weight: {malicious_mean:.4f}")
        assert malicious_mean < honest_mean, "Malicious clients should have lower weights"
        
        # 4. Check weight distribution
        print("\nWeight distribution analysis:")
        for attack_type in ['honest', 'sign_flip', 'scale', 'pattern', 'random']:
            type_weights = [w.item() for w, t in zip(weights.detach(), attack_types) if t == attack_type]
            if type_weights:
                print(f"\n{attack_type.upper()}:")
                print(f"  Min weight: {min(type_weights):.4f}")
                print(f"  Max weight: {max(type_weights):.4f}")
                print(f"  Mean weight: {np.mean(type_weights):.4f}")
        
        # 5. Verify attack-specific behavior
        print("\nVerifying attack-specific behavior:")
        
        # Sign-flip attacks should have very low weights
        sign_flip_weights = [w.item() for w, t in zip(weights.detach(), attack_types) if t == 'sign_flip']
        assert np.mean(sign_flip_weights) < 0.1, "Sign-flip attacks should have very low weights"
        
        # Scale attacks should have low weights
        scale_weights = [w.item() for w, t in zip(weights.detach(), attack_types) if t == 'scale']
        assert np.mean(scale_weights) < 0.2, "Scale attacks should have low weights"
        
        # Pattern attacks should have low weights
        pattern_weights = [w.item() for w, t in zip(weights.detach(), attack_types) if t == 'pattern']
        assert np.mean(pattern_weights) < 0.1, "Pattern attacks should have low weights"
        
        # Random attacks should have very low weights
        random_weights = [w.item() for w, t in zip(weights.detach(), attack_types) if t == 'random']
        assert np.mean(random_weights) < 0.1, "Random attacks should have very low weights"
        
        print("\n✓ All tests passed successfully!")
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        print("Starting gradient weighting test...")
        import torch
        print("PyTorch imported successfully")
        import numpy as np
        print("NumPy imported successfully")
        from federated_learning.models.attention import DualAttention
        print("DualAttention imported successfully")
        from federated_learning.training.aggregation import extract_gradient_features
        print("extract_gradient_features imported successfully")
        
        success = test_gradient_weighting()
        print(f"\nTest {'passed' if success else 'failed'}")
    except ImportError as e:
        print(f"Import error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
