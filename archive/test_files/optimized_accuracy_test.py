#!/usr/bin/env python3
"""
🎯 OPTIMIZED ACCURACY VALIDATION TEST

This script tests the improved configuration to validate that we can achieve
realistic accuracy results before running full experiments.

Expected Results:
- CIFAR-10 + ResNet18: 80%+ accuracy (target: 85%+)
- MNIST + CNN: 95%+ accuracy (target: 98%+)
- Alzheimer + ResNet18: 90%+ accuracy (target: 95%+)
"""

import sys
import os
import torch
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_configuration(dataset='CIFAR10', quick_test=True):
    """Test specific dataset configuration."""
    
    print(f"\n🧪 TESTING {dataset} CONFIGURATION")
    print("=" * 60)
    
    # Import optimized config
    print("Loading optimized configuration...")
    
    # Override config file temporarily
    config_override = f"""
# Override config.py to use optimized settings
import sys
sys.path.insert(0, 'federated_learning/config')
from config_optimized import *

# Quick test overrides
if {quick_test}:
    GLOBAL_EPOCHS = 8          # Quick test: 8 epochs
    LOCAL_EPOCHS_ROOT = 8      # Quick test: 8 pretraining epochs
    VAE_EPOCHS = 8             # Quick test: 8 VAE epochs
    print("⚡ Quick test mode: Reduced epochs for faster validation")
else:
    print("🏁 Full test mode: Using optimized epochs for final results")

# Dataset-specific optimizations
DATASET = '{dataset}'
if DATASET == 'CIFAR10':
    MODEL = 'ResNet18'
    INPUT_CHANNELS = 3
    NUM_CLASSES = 10
    EXPECTED_ACCURACY = 0.80 if {quick_test} else 0.85
elif DATASET == 'MNIST':
    MODEL = 'CNN'
    INPUT_CHANNELS = 1
    NUM_CLASSES = 10
    EXPECTED_ACCURACY = 0.95 if {quick_test} else 0.98
elif DATASET == 'ALZHEIMER':
    MODEL = 'ResNet18'
    INPUT_CHANNELS = 3
    NUM_CLASSES = 4
    EXPECTED_ACCURACY = 0.90 if {quick_test} else 0.95

print(f"🎯 Target accuracy for {{DATASET}}: {{EXPECTED_ACCURACY:.1%}}")
"""
    
    # Write temporary config override
    with open('temp_config_test.py', 'w') as f:
        f.write(config_override)
    
    try:
        # Import the configuration
        exec(open('temp_config_test.py').read())
        
        # Import federated learning components
        from federated_learning.training.server import Server
        from federated_learning.training.client import Client
        from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
        from federated_learning.utils.model_utils import set_random_seeds
        
        # Set random seed
        if 'RANDOM_SEED' in locals():
            set_random_seeds(RANDOM_SEED)
        
        print(f"\n📊 Configuration Summary:")
        print(f"   Dataset: {locals().get('DATASET', 'Unknown')}")
        print(f"   Model: {locals().get('MODEL', 'Unknown')}")
        print(f"   Global Epochs: {locals().get('GLOBAL_EPOCHS', 'Unknown')}")
        print(f"   Batch Size: {locals().get('BATCH_SIZE', 'Unknown')}")
        print(f"   Learning Rate: {locals().get('LR', 'Unknown')}")
        print(f"   Root Dataset Size: {locals().get('ROOT_DATASET_SIZE', 'Unknown')}")
        
        # Load dataset
        print(f"\n📂 Loading {dataset} dataset...")
        start_time = time.time()
        
        root_dataset, test_dataset = load_dataset()
        print(f"   ✅ Dataset loaded in {time.time() - start_time:.2f}s")
        print(f"   📊 Training samples: {len(root_dataset)}")
        print(f"   📊 Test samples: {len(test_dataset)}")
        
        # Create data loaders
        root_loader = torch.utils.data.DataLoader(
            root_dataset, 
            batch_size=locals().get('BATCH_SIZE', 32), 
            shuffle=True, 
            num_workers=0
        )
        
        # Create server
        print(f"\n🖥️ Creating server...")
        server = Server()
        server.set_datasets(root_loader, test_dataset)
        
        # Get initial accuracy
        print(f"\n📈 Evaluating initial model...")
        initial_accuracy = server.evaluate_model()
        print(f"   📊 Initial accuracy: {initial_accuracy:.4f} ({initial_accuracy:.2%})")
        
        # Pre-train global model
        print(f"\n🔧 Pre-training global model...")
        start_training = time.time()
        server._pretrain_global_model()
        
        # Evaluate after pretraining
        pretrained_accuracy = server.evaluate_model()
        training_time = time.time() - start_training
        
        print(f"   ✅ Pre-training completed in {training_time:.2f}s")
        print(f"   📊 Pre-trained accuracy: {pretrained_accuracy:.4f} ({pretrained_accuracy:.2%})")
        
        # Check if we meet expectations
        expected_acc = locals().get('EXPECTED_ACCURACY', 0.8)
        
        if pretrained_accuracy >= expected_acc:
            print(f"   🎉 SUCCESS: Accuracy {pretrained_accuracy:.2%} meets target {expected_acc:.2%}")
            status = "✅ PASSED"
        elif pretrained_accuracy >= expected_acc * 0.9:
            print(f"   ⚠️  CLOSE: Accuracy {pretrained_accuracy:.2%} is close to target {expected_acc:.2%}")
            status = "⚠️ CLOSE"
        else:
            print(f"   ❌ FAILED: Accuracy {pretrained_accuracy:.2%} below target {expected_acc:.2%}")
            status = "❌ FAILED"
        
        # Calculate improvement
        improvement = pretrained_accuracy - initial_accuracy
        print(f"   📈 Improvement: {improvement:.4f} ({improvement:.2%})")
        
        return {
            'dataset': dataset,
            'initial_accuracy': initial_accuracy,
            'final_accuracy': pretrained_accuracy,
            'improvement': improvement,
            'training_time': training_time,
            'expected_accuracy': expected_acc,
            'status': status,
            'passed': pretrained_accuracy >= expected_acc
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset,
            'status': '❌ ERROR',
            'error': str(e),
            'passed': False
        }
    
    finally:
        # Clean up
        if os.path.exists('temp_config_test.py'):
            os.remove('temp_config_test.py')

def run_comprehensive_validation():
    """Run validation tests for all datasets."""
    
    print("🚀 COMPREHENSIVE ACCURACY VALIDATION")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test datasets in order of complexity
    datasets = ['CIFAR10']  # Start with most problematic
    
    results = []
    total_start = time.time()
    
    for dataset in datasets:
        print(f"\n{'='*20} {dataset} TEST {'='*20}")
        result = test_configuration(dataset, quick_test=True)
        results.append(result)
        
        # Brief pause between tests
        time.sleep(2)
    
    total_time = time.time() - total_start
    
    # Summary report
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 80)
    
    passed_count = 0
    for result in results:
        dataset = result['dataset']
        status = result['status']
        
        if 'final_accuracy' in result:
            acc = result['final_accuracy']
            expected = result['expected_accuracy']
            print(f"   {dataset:12} | {status:12} | {acc:.2%} (target: {expected:.2%})")
        else:
            print(f"   {dataset:12} | {status:12}")
        
        if result.get('passed', False):
            passed_count += 1
    
    print(f"\n🎯 Results: {passed_count}/{len(results)} tests passed")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    
    if passed_count == len(results):
        print(f"\n🎉 ALL TESTS PASSED! Configuration is ready for full experiments.")
        recommendation = "✅ Proceed with full experiments using optimized config"
    elif passed_count > 0:
        print(f"\n⚠️ PARTIAL SUCCESS: Some tests passed. Review failed cases.")
        recommendation = "⚠️ Review failures, consider parameter adjustments"
    else:
        print(f"\n❌ ALL TESTS FAILED: Configuration needs significant adjustment.")
        recommendation = "❌ Major configuration changes needed"
    
    print(f"\n💡 Recommendation: {recommendation}")
    
    return results

if __name__ == "__main__":
    print("🧪 OPTIMIZED ACCURACY VALIDATION TEST")
    print(f"🎯 Objective: Validate improved training parameters")
    print(f"⚡ Mode: Quick validation (8 epochs)")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run validation
    results = run_comprehensive_validation()
    
    print(f"\n✅ Validation completed!")
    print(f"💡 Next steps:")
    print(f"   1. If tests pass → Run full experiments with main.py")
    print(f"   2. If tests fail → Adjust config_optimized.py parameters")
    print(f"   3. Document final results in paper tables") 