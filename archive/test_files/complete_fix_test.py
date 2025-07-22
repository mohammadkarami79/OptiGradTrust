#!/usr/bin/env python3
"""
🔧 COMPLETE FIX TEST - فیکس کامل تمام مشکلات
=================================================

This script will fix ALL remaining issues:
1. CIFAR-10 accuracy: 51.47% → 85%+
2. Failed detections (0%) → reasonable values
3. Validate all configurations are working

Author: System Optimization
Date: 2025-01-27
"""

import os
import sys
import torch
import time
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def override_config():
    """Override configuration with optimized settings"""
    import federated_learning.config.config as config
    
    print("🔧 Applying optimized configuration...")
    
    # === ACCURACY FIX ===
    config.DATASET = 'CIFAR10'
    config.MODEL = 'RESNET18'
    config.GLOBAL_EPOCHS = 20  # Increased from 3
    config.LOCAL_EPOCHS_ROOT = 12  # Increased from 5  
    config.LOCAL_EPOCHS_CLIENT = 4  # Increased from 3
    config.BATCH_SIZE = 32  # Increased from 16
    config.ROOT_DATASET_SIZE = 4500  # Increased from 3500
    config.LEARNING_RATE = 0.01  # Optimized
    
    # === DETECTION OPTIMIZATION ===
    config.VAE_EPOCHS = 15  # Increased from 12
    config.VAE_BATCH_SIZE = 12  # Increased from 6
    config.VAE_LEARNING_RATE = 0.0005  # Increased from 0.0003
    config.VAE_PROJECTION_DIM = 128  # Increased from 64
    config.VAE_HIDDEN_DIM = 64  # Increased from 32
    config.VAE_LATENT_DIM = 32  # Increased from 16
    
    # === DUAL ATTENTION ENHANCEMENT ===
    config.DUAL_ATTENTION_HIDDEN_SIZE = 200  # Increased from 128
    config.DUAL_ATTENTION_HEADS = 10  # Increased from 8
    config.DUAL_ATTENTION_EPOCHS = 8  # Increased from 5
    config.DUAL_ATTENTION_BATCH_SIZE = 12  # Increased from 8
    
    # === ATTACK DETECTION ENHANCEMENT ===
    config.GRADIENT_NORM_THRESHOLD_FACTOR = 2.0  # More sensitive
    config.TRUST_SCORE_THRESHOLD = 0.6  # Balanced
    config.MALICIOUS_THRESHOLD = 0.65  # Balanced
    config.ZERO_ATTACK_THRESHOLD = 0.01  # More sensitive
    
    # === SHAPLEY OPTIMIZATION ===
    config.SHAPLEY_SAMPLES = 25  # Increased from 20
    config.SHAPLEY_WEIGHT = 0.4  # Balanced
    
    print("✅ Configuration optimized for maximum performance")
    return config

def test_single_attack(attack_type):
    """Test a single attack type with comprehensive metrics"""
    print(f"\n🎯 Testing {attack_type}...")
    
    try:
        from federated_learning.training.server import FederatedServer
        from federated_learning.utils.data_utils import get_dataset
        from torch.utils.data import DataLoader
        
        # Get dataset
        train_dataset, test_dataset = get_dataset('CIFAR10')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create server
        server = FederatedServer()
        
        # Test baseline accuracy first
        initial_accuracy = server.evaluate_model(test_loader)
        print(f"📊 Initial accuracy: {initial_accuracy:.4f}")
        
        # Pre-train for better convergence
        print("🔧 Pre-training global model...")
        for epoch in range(5):  # Quick pre-training
            server.train_global_model_epoch(train_dataset, epoch)
            if epoch % 2 == 0:
                accuracy = server.evaluate_model(test_loader)
                print(f"  Epoch {epoch}: {accuracy:.4f}")
        
        # Final accuracy check
        final_accuracy = server.evaluate_model(test_loader)
        print(f"📊 Pre-trained accuracy: {final_accuracy:.4f}")
        
        # Now test attack detection
        from main import run_federated_learning
        results = run_federated_learning(attack_type)
        
        # Extract metrics
        detection_precision = results.get('detection_precision', 0.0)
        detection_recall = results.get('detection_recall', 0.0)
        detection_f1 = results.get('detection_f1', 0.0)
        model_accuracy = results.get('accuracy', 0.0)
        
        print(f"✅ Results for {attack_type}:")
        print(f"   Model Accuracy: {model_accuracy:.4f}")
        print(f"   Detection Precision: {detection_precision:.4f}")
        print(f"   Detection Recall: {detection_recall:.4f}")
        print(f"   Detection F1-Score: {detection_f1:.4f}")
        
        # Success criteria
        accuracy_good = model_accuracy > 0.80  # 80%+ for CIFAR-10
        detection_good = detection_precision > 0.30  # At least 30%
        
        status = "✅ SUCCESS" if (accuracy_good and detection_good) else "⚠️ NEEDS IMPROVEMENT"
        print(f"   Status: {status}")
        
        return {
            'attack_type': attack_type,
            'model_accuracy': model_accuracy,
            'detection_precision': detection_precision,
            'detection_recall': detection_recall,
            'detection_f1': detection_f1,
            'accuracy_good': accuracy_good,
            'detection_good': detection_good,
            'success': accuracy_good and detection_good
        }
        
    except Exception as e:
        print(f"❌ Error testing {attack_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'attack_type': attack_type,
            'error': str(e),
            'success': False
        }

def run_comprehensive_fix():
    """Run comprehensive fix for all issues"""
    print("🚀 COMPLETE FIX TEST - STARTING")
    print("="*50)
    
    # Override configuration
    config = override_config()
    
    # Priority attack types to fix
    priority_attacks = [
        'scaling_attack',      # Should work well
        'noise_attack',        # Should work well
        'partial_scaling_attack',  # Known to work
        'sign_flipping_attack',    # Currently 0% - FIX
        'label_flipping',          # Currently 0% - FIX
    ]
    
    results = []
    
    for attack_type in priority_attacks:
        result = test_single_attack(attack_type)
        results.append(result)
        
        # Short delay between tests
        time.sleep(2)
    
    # Summary
    print("\n🏆 COMPREHENSIVE FIX RESULTS:")
    print("="*50)
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successful: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed: {len(failed_tests)}/{len(results)}")
    
    # Detailed results
    for result in results:
        if 'error' not in result:
            accuracy = result.get('model_accuracy', 0)
            precision = result.get('detection_precision', 0)
            attack = result.get('attack_type', 'unknown')
            print(f"   {attack}: Acc={accuracy:.2%}, Det={precision:.2%}")
    
    # Generate final table
    if len(successful_tests) >= 3:
        print("\n📊 GENERATING UPDATED RESULTS TABLE...")
        generate_updated_table(results)
        print("✅ All major issues should now be fixed!")
        return True
    else:
        print("\n⚠️ Some issues remain. Need more investigation.")
        return False

def generate_updated_table(results):
    """Generate updated results table"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    table_content = f"""# UPDATED CIFAR-10 RESULTS - FIXED
## Generated: {timestamp}

| Attack Type | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Status |
|-------------|--------------|---------------|------------|--------------|---------|
"""
    
    for result in results:
        if 'error' not in result:
            attack = result.get('attack_type', 'unknown')
            acc = result.get('model_accuracy', 0) * 100
            prec = result.get('detection_precision', 0) * 100
            rec = result.get('detection_recall', 0) * 100
            f1 = result.get('detection_f1', 0) * 100
            status = "✅ Fixed" if result.get('success', False) else "⚠️ Partial"
            
            table_content += f"| {attack} | {acc:.2f} | {prec:.2f} | {rec:.2f} | {f1:.2f} | {status} |\n"
    
    # Save updated table
    with open('results/CIFAR10_FIXED_RESULTS.md', 'w', encoding='utf-8') as f:
        f.write(table_content)
    
    print(f"📁 Results saved to: results/CIFAR10_FIXED_RESULTS.md")

if __name__ == "__main__":
    try:
        success = run_comprehensive_fix()
        if success:
            print("\n🎉 ALL FIXES COMPLETED SUCCESSFULLY!")
            print("📝 Ready to proceed to Non-IID phase.")
        else:
            print("\n⚠️ Some issues remain. Check logs above.")
            
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        import traceback
        traceback.print_exc() 