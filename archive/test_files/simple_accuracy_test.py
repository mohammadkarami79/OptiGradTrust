#!/usr/bin/env python3
"""
🎯 SIMPLE ACCURACY TEST

Fast validation test to check if optimized configuration 
can achieve 80%+ accuracy for CIFAR-10
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_simple_test():
    """Run a simple accuracy test with optimized configuration."""
    
    print("🧪 SIMPLE ACCURACY VALIDATION TEST")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🎯 Target: 80%+ CIFAR-10 accuracy")
    
    try:
        # Import simple optimized config
        print("\n📋 Loading optimized configuration...")
        import simple_optimized_config as config
        print(f"   ✅ Config loaded: {config.DATASET} + {config.MODEL}")
        print(f"   📊 Training epochs: {config.GLOBAL_EPOCHS}")
        print(f"   📊 Batch size: {config.BATCH_SIZE}")
        print(f"   📊 Learning rate: {config.LR}")
        
        # CRITICAL FIX: Override the federated learning config
        print("\n🔧 Overriding federated learning config...")
        import federated_learning.config.config as fl_config
        
        # Copy all config values from our optimized config to fl_config
        for attr in dir(config):
            if not attr.startswith('_'):  # Skip private attributes
                setattr(fl_config, attr, getattr(config, attr))
        
        print(f"   ✅ Config override complete")
        print(f"   📊 FL Config now uses: {fl_config.DATASET} + {fl_config.MODEL}")
        
        # Import federated learning components AFTER config override
        print("   Loading federated learning modules...")
        from federated_learning.training.server import Server
        from federated_learning.data.dataset_utils import load_dataset
        from federated_learning.utils.model_utils import set_random_seeds
        
        # Set random seed
        set_random_seeds(config.RANDOM_SEED)
        print(f"   ✅ Random seed set: {config.RANDOM_SEED}")
        
        # Load dataset
        print(f"\n📂 Loading {config.DATASET} dataset...")
        start_time = time.time()
        
        root_dataset, test_dataset = load_dataset()
        load_time = time.time() - start_time
        
        print(f"   ✅ Dataset loaded in {load_time:.2f}s")
        print(f"   📊 Training samples: {len(root_dataset)}")
        print(f"   📊 Test samples: {len(test_dataset)}")
        
        # Create data loader
        import torch
        root_loader = torch.utils.data.DataLoader(
            root_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0
        )
        
        # Create server
        print(f"\n🖥️ Creating server...")
        server = Server()
        server.set_datasets(root_loader, test_dataset)
        print(f"   ✅ Server created with {config.MODEL}")
        
        # Evaluate initial model
        print(f"\n📈 Evaluating initial model...")
        initial_accuracy = server.evaluate_model()
        print(f"   📊 Initial accuracy: {initial_accuracy:.4f} ({initial_accuracy:.2%})")
        
        # Pre-train global model (KEY STEP)
        print(f"\n🔧 Pre-training global model for {config.LOCAL_EPOCHS_ROOT} epochs...")
        print(f"   This may take 3-5 minutes for {config.LOCAL_EPOCHS_ROOT} epochs...")
        start_training = time.time()
        
        server._pretrain_global_model()
        
        training_time = time.time() - start_training
        print(f"   ✅ Pre-training completed in {training_time:.2f}s")
        
        # Evaluate after pre-training
        print(f"\n📊 Evaluating pre-trained model...")
        final_accuracy = server.evaluate_model()
        improvement = final_accuracy - initial_accuracy
        
        print(f"   📊 Final accuracy: {final_accuracy:.4f} ({final_accuracy:.2%})")
        print(f"   📈 Improvement: {improvement:.4f} ({improvement:.2%})")
        
        # Check target achievement
        target_accuracy = 0.80  # 80% target
        
        print(f"\n🎯 TARGET ASSESSMENT:")
        print(f"   Target accuracy: {target_accuracy:.2%}")
        print(f"   Achieved accuracy: {final_accuracy:.2%}")
        
        if final_accuracy >= target_accuracy:
            status = "🎉 SUCCESS"
            print(f"   {status}: Target achieved!")
        elif final_accuracy >= target_accuracy * 0.9:  # 72%+
            status = "⚠️ CLOSE"
            print(f"   {status}: Close to target (90%+ of goal)")
        else:
            status = "❌ FAILED"
            print(f"   {status}: Below target threshold")
        
        # Summary
        total_time = load_time + training_time
        print(f"\n📋 SUMMARY:")
        print(f"   Status: {status}")
        print(f"   Dataset: {config.DATASET}")
        print(f"   Model: {config.MODEL}")
        print(f"   Initial → Final: {initial_accuracy:.2%} → {final_accuracy:.2%}")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        
        # Performance analysis
        print(f"\n🔍 PERFORMANCE ANALYSIS:")
        if final_accuracy >= 0.85:
            print(f"   🌟 EXCELLENT: Exceeds research standards")
        elif final_accuracy >= 0.80:
            print(f"   ✅ GOOD: Meets research standards")
        elif final_accuracy >= 0.70:
            print(f"   ⚠️ ACCEPTABLE: Reasonable for initial test")
        elif final_accuracy >= 0.60:
            print(f"   🔧 NEEDS_WORK: Requires parameter tuning")
        else:
            print(f"   ❌ POOR: Major configuration issues")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if final_accuracy >= target_accuracy:
            print(f"   ✅ Configuration is ready for full experiments")
            print(f"   ✅ Proceed with complete attack detection tests")
            print(f"   ✅ Expected paper results: CIFAR-10 accuracy 85-90%")
        elif final_accuracy >= 0.70:  # 70%+
            print(f"   ⚠️ Good progress, consider minor tweaks:")
            print(f"   ⚠️ - Increase GLOBAL_EPOCHS to 25-30")
            print(f"   ⚠️ - Increase LOCAL_EPOCHS_ROOT to 15-20")
        else:
            print(f"   ❌ Major improvements needed:")
            print(f"   ❌ - Increase all epoch counts significantly")
            print(f"   ❌ - Consider larger batch sizes if memory allows")
        
        return {
            'status': status,
            'initial_accuracy': initial_accuracy,
            'final_accuracy': final_accuracy,
            'improvement': improvement,
            'training_time': training_time,
            'total_time': total_time,
            'target_met': final_accuracy >= target_accuracy
        }
        
    except Exception as e:
        print(f"\n❌ ERROR OCCURRED:")
        print(f"   Error: {str(e)}")
        
        # Try to provide helpful debug info
        import traceback
        print(f"\n🔍 DEBUG INFO:")
        traceback.print_exc()
        
        return {
            'status': '❌ ERROR',
            'error': str(e),
            'target_met': False
        }

if __name__ == "__main__":
    print("🚀 Starting simple accuracy validation...")
    print("🔧 This test validates optimized configuration for CIFAR-10")
    print("⏱️ Expected runtime: 3-5 minutes")
    
    result = run_simple_test()
    
    print(f"\n" + "="*60)
    print(f"✅ TEST COMPLETED!")
    
    if result.get('target_met', False):
        print(f"🎉 SUCCESS: Ready for full experiments!")
        print(f"📝 Next step: Run main.py with optimized config")
    elif result.get('status') == '⚠️ CLOSE':
        print(f"⚠️ CLOSE: Minor adjustments recommended")
        print(f"📝 Next step: Increase epoch counts slightly")
    else:
        print(f"🔧 NEEDS WORK: Consider parameter adjustments")
        print(f"📝 Next step: Review and optimize configuration")
    
    print(f"="*60) 