#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.utils.attack_utils import apply_attack, ScalingAttack, PartialScalingAttack
from federated_learning.training.client import Client

def test_attack_interface():
    """Test that attacks now work with both interfaces"""
    
    # Create a dummy client
    class DummyClient:
        def __init__(self):
            self.client_id = 1
            self.is_malicious = False
    
    client = DummyClient()
    
    # Test ScalingAttack directly
    print("=== Testing ScalingAttack Interface ===")
    attack = ScalingAttack(scale_factor=5.0)
    
    # Create test gradient
    gradient = torch.randn(100) * 0.1
    original_norm = torch.norm(gradient).item()
    
    print(f"Original gradient norm: {original_norm:.4f}")
    
    # Test apply() method
    attacked_grad_1 = attack.apply(gradient)
    norm_1 = torch.norm(attacked_grad_1).item()
    print(f"After apply(): {norm_1:.4f} (expected ~{original_norm * 5:.4f})")
    
    # Test apply_gradient_attack() method
    attacked_grad_2 = attack.apply_gradient_attack(gradient)
    norm_2 = torch.norm(attacked_grad_2).item()
    print(f"After apply_gradient_attack(): {norm_2:.4f}")
    
    # Verify they give same result
    diff = torch.norm(attacked_grad_1 - attacked_grad_2).item()
    print(f"Difference between methods: {diff:.8f}")
    
    # Test apply_attack function
    print("\n=== Testing apply_attack Function ===")
    apply_attack(client, 'scaling_attack')
    
    if hasattr(client, 'attack'):
        print("✅ Attack successfully applied to client")
        test_grad = torch.randn(50) * 0.05
        result = client.attack.apply_gradient_attack(test_grad)
        print(f"✅ apply_gradient_attack() works: {torch.norm(result).item():.4f}")
    else:
        print("❌ Attack not applied to client")
    
    print("\n=== Testing PartialScalingAttack ===")
    partial_attack = PartialScalingAttack(scale_factor=10.0, fraction=0.3)
    test_grad = torch.randn(1000) * 0.1
    original_norm = torch.norm(test_grad).item()
    attacked = partial_attack.apply_gradient_attack(test_grad)
    attacked_norm = torch.norm(attacked).item()
    
    print(f"Partial scaling: {original_norm:.4f} → {attacked_norm:.4f}")
    print(f"Increase factor: {attacked_norm/original_norm:.2f}x")
    
    if attacked_norm > original_norm * 1.5:  # Should be significantly larger
        print("✅ PartialScalingAttack working correctly")
    else:
        print("❌ PartialScalingAttack not working as expected")

if __name__ == "__main__":
    test_attack_interface() 