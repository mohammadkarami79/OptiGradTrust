#!/usr/bin/env python3
"""
Quick MNIST Attack Test - یک round سریع برای تایید نتایج

Test all 5 attack types on MNIST to verify actual precision/recall
This is for validation before reporting final numbers.
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import config with MNIST settings
from federated_learning.config.config import *

# Set config for MNIST + fast testing
DATASET = 'MNIST'
MODEL = 'CNN'
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.3
GLOBAL_EPOCHS = 3  # سریع برای تست
LOCAL_EPOCHS_CLIENT = 3
VAE_EPOCHS = 15

def quick_test_attack(attack_type):
    """Test one attack type quickly"""
    print(f"\n🧪 Testing {attack_type}...")
    
    try:
        # Clear main module
        if 'main' in sys.modules:
            del sys.modules['main']
        
        # Set attack type
        sys.path.insert(0, 'federated_learning/config/')
        
        # Create a temporary config override
        config_override = f"""
# Quick test override for {attack_type}
DATASET = 'MNIST'
MODEL = 'CNN'
ENABLE_ATTACK_SIMULATION = True
ATTACK_TYPE = '{attack_type}'
GLOBAL_EPOCHS = 3
LOCAL_EPOCHS_CLIENT = 3
VAE_EPOCHS = 15
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.3
ENABLE_DETECTION = True
ENABLE_VAE_ANOMALY_DETECTION = True
ENABLE_SHAPLEY_DETECTION = True
ENABLE_DUAL_ATTENTION = True
"""
        
        # Write temporary config
        with open('temp_test_config.py', 'w') as f:
            f.write(config_override)
        
        # Import updated config
        import temp_test_config
        
        # Update config module
        import federated_learning.config.config as cfg
        for attr_name in dir(temp_test_config):
            if not attr_name.startswith('_'):
                setattr(cfg, attr_name, getattr(temp_test_config, attr_name))
        
        # Run main with this attack
        from main import main
        result = main()
        
        # Extract key metrics
        final_acc = result.get('final_accuracy', 0)
        det_metrics = result.get('detection_metrics', {})
        precision = det_metrics.get('precision', 0)
        recall = det_metrics.get('recall', 0)
        f1 = det_metrics.get('f1_score', 0)
        
        print(f"✅ {attack_type}: Acc={final_acc:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Clean up
        if os.path.exists('temp_test_config.py'):
            os.remove('temp_test_config.py')
        
        return {
            'attack': attack_type,
            'accuracy': final_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ {attack_type} failed: {str(e)}")
        return {
            'attack': attack_type,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Run quick test for all 5 attacks"""
    print("🚀 QUICK MNIST ATTACK VALIDATION")
    print("="*50)
    
    # 5 main attack types
    attack_types = [
        'scaling_attack',
        'partial_scaling_attack', 
        'sign_flipping_attack',
        'noise_attack',
        'label_flipping'
    ]
    
    results = []
    
    for attack in attack_types:
        result = quick_test_attack(attack)
        results.append(result)
    
    # Print summary table
    print(f"\n📊 MNIST QUICK TEST RESULTS SUMMARY:")
    print("="*60)
    print(f"{'Attack':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Status':<10}")
    print("-"*60)
    
    for r in results:
        if r['status'] == 'success':
            print(f"{r['attack']:<20} {r['accuracy']:<10.3f} {r['precision']:<10.3f} {r['recall']:<8.3f} {r['f1_score']:<8.3f} {'✅':<10}")
        else:
            print(f"{r['attack']:<20} {'Failed':<10} {'Failed':<10} {'Failed':<8} {'Failed':<8} {'❌':<10}")
    
    # Calculate averages for successful tests
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        avg_acc = sum(r['accuracy'] for r in successful) / len(successful)
        avg_prec = sum(r['precision'] for r in successful) / len(successful)
        avg_recall = sum(r['recall'] for r in successful) / len(successful)
        avg_f1 = sum(r['f1_score'] for r in successful) / len(successful)
        
        print("-"*60)
        print(f"{'AVERAGE':<20} {avg_acc:<10.3f} {avg_prec:<10.3f} {avg_recall:<8.3f} {avg_f1:<8.3f} {'📊':<10}")
    
    print(f"\n✅ Quick validation completed: {len(successful)}/{len(attack_types)} attacks successful")
    
    return results

if __name__ == "__main__":
    main() 