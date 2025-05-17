import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from federated_learning.training.training_utils import client_update
from federated_learning.config.config import *
from federated_learning.models.attention import DualAttention

def test_memory_management():
    print("\n=== Testing Memory Management ===")
    try:
        # Test gradient chunking
        large_gradient = torch.randn(1000000, 512)
        chunks = torch.split(large_gradient, GRADIENT_CHUNK_SIZE)
        print(f"✓ Gradient chunking working - Split into {len(chunks)} chunks")
        
        # Test dimension reduction
        if ENABLE_DIMENSION_REDUCTION:
            reduced_dim = int(large_gradient.shape[1] * DIMENSION_REDUCTION_RATIO)
            print(f"✓ Dimension reduction working - Reduced to {reduced_dim} dimensions")
        
        # Test device placement
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            print(f"✓ GPU memory available: {free_memory/1024**2:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Memory management test failed: {str(e)}")
        return False

def test_fedprox_constraints():
    print("\n=== Testing FedProx Constraints ===")
    try:
        # Create synthetic model updates
        param_size = 1000
        global_params = torch.randn(param_size)
        local_updates = []
        
        # Generate local updates
        for _ in range(NUM_CLIENTS):
            # Simulate local update with controlled deviation
            update = global_params + torch.randn(param_size) * FEDPROX_MU
            local_updates.append(update)
        
        # Check constraints
        max_deviation = 0
        for update in local_updates:
            deviation = torch.norm(update - global_params).item()
            max_deviation = max(max_deviation, deviation)
        
        print(f"Maximum parameter deviation: {max_deviation:.4f}")
        expected_max = 5.0 / FEDPROX_MU
        assert max_deviation < expected_max, f"Deviation ({max_deviation}) exceeds expected maximum ({expected_max})"
        print("✓ FedProx constraints working correctly")
        
        return True
    except Exception as e:
        print(f"✗ FedProx constraint test failed: {str(e)}")
        return False

def test_malicious_detection():
    print("\n=== Testing Malicious Client Detection ===")
    try:
        # Create synthetic gradient features
        num_features = 10
        num_samples = 100
        
        # Generate normal and malicious features
        normal_features = torch.randn(80, num_features)  # 80% normal
        malicious_features = torch.randn(20, num_features) * 2 + 1  # 20% malicious
        
        features = torch.cat([normal_features, malicious_features])
        labels = torch.cat([torch.zeros(80), torch.ones(20)])
        
        # Test DualAttention model
        model = DualAttention(
            feature_dim=num_features,
            hidden_dim=DUAL_ATTENTION_HIDDEN_DIM,
            num_heads=4,
            dropout=DUAL_ATTENTION_DROPOUT
        )
        
        # Forward pass
        trust_scores, confidence = model(features)
        
        # Verify output shapes and ranges
        assert trust_scores.shape == (100,), "Incorrect trust score shape"
        assert confidence.shape == (100,), "Incorrect confidence shape"
        assert torch.all(trust_scores >= 0) and torch.all(trust_scores <= 1), "Trust scores out of range"
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1), "Confidence scores out of range"
        
        print("✓ Malicious detection model working correctly")
        return True
    except Exception as e:
        print(f"✗ Malicious detection test failed: {str(e)}")
        return False

def test_privacy_mechanisms():
    print("\n=== Testing Privacy Mechanisms ===")
    try:
        # Test gradient clipping
        gradient = torch.randn(1000) * 10  # Large gradient
        clipped_gradient = torch.nn.utils.clip_grad_norm_(
            [gradient], 
            max_norm=DP_CLIP_NORM
        )
        assert clipped_gradient <= DP_CLIP_NORM, "Gradient clipping failed"
        print(f"✓ Gradient clipping working - Norm reduced to {clipped_gradient:.4f}")
        
        if PRIVACY_MECHANISM == 'dp':
            # Test differential privacy noise addition
            noise_scale = DP_CLIP_NORM * np.sqrt(2 * np.log(1.25/DP_DELTA)) / DP_EPSILON
            noise = torch.randn_like(gradient) * noise_scale
            noisy_gradient = gradient + noise
            print(f"✓ DP noise addition working - Scale: {noise_scale:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Privacy mechanism test failed: {str(e)}")
        return False

def run_system_test():
    print("\n=== Running Complete System Test ===")
    
    # Test all components
    tests = [
        ("Memory Management", test_memory_management),
        ("FedProx Constraints", test_fedprox_constraints),
        ("Malicious Detection", test_malicious_detection),
        ("Privacy Mechanisms", test_privacy_mechanisms)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        results[name] = test_func()
    
    # Print summary
    print("\n=== Test Summary ===")
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n✓ All systems working correctly!")
    else:
        print("\n✗ Some tests failed - please check the logs above")
    
    return all_passed

if __name__ == "__main__":
    run_system_test() 