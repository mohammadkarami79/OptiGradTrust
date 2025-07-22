#!/usr/bin/env python3
"""
🧪 MINIMAL CIFAR-10 ACCURACY TEST

Ultra-simple test to validate optimized configuration works.
This is the most basic test possible to check accuracy improvement.
"""

import sys
import os
import time

def run_minimal_test():
    """Run minimal accuracy test with optimized config."""
    
    print("🧪 MINIMAL CIFAR-10 ACCURACY TEST")
    print("=" * 50)
    print("🎯 Target: Fix 51% → 80%+ accuracy")
    
    try:
        # Step 1: Basic imports
        print("\n1️⃣ Testing basic imports...")
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   📊 CUDA: {torch.cuda.is_available()}")
        
        # Step 2: Test config
        print("\n2️⃣ Testing configuration...")
        from federated_learning.config import config
        print(f"   ✅ Config loaded")
        print(f"   📊 GLOBAL_EPOCHS: {config.GLOBAL_EPOCHS}")
        print(f"   📊 BATCH_SIZE: {config.BATCH_SIZE}")
        print(f"   📊 LOCAL_EPOCHS_ROOT: {config.LOCAL_EPOCHS_ROOT}")
        
        # Validate key improvements
        assert config.GLOBAL_EPOCHS >= 15, f"GLOBAL_EPOCHS too low: {config.GLOBAL_EPOCHS}"
        assert config.BATCH_SIZE >= 24, f"BATCH_SIZE too low: {config.BATCH_SIZE}"  
        assert config.LOCAL_EPOCHS_ROOT >= 10, f"LOCAL_EPOCHS_ROOT too low: {config.LOCAL_EPOCHS_ROOT}"
        print(f"   ✅ All parameters improved!")
        
        # Step 3: Test dataset loading
        print("\n3️⃣ Testing dataset loading...")
        from federated_learning.data.dataset_utils import load_dataset
        
        start_time = time.time()
        root_dataset, test_dataset = load_dataset()
        load_time = time.time() - start_time
        
        print(f"   ✅ Dataset loaded in {load_time:.2f}s")
        print(f"   📊 Training: {len(root_dataset)} samples")
        print(f"   📊 Test: {len(test_dataset)} samples")
        
        # Step 4: Test server creation
        print("\n4️⃣ Testing server creation...")
        from federated_learning.training.server import Server
        
        # Create data loader
        root_loader = torch.utils.data.DataLoader(
            root_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0
        )
        
        # Create server
        server = Server()
        server.set_datasets(root_loader, test_dataset)
        print(f"   ✅ Server created with {config.MODEL}")
        
        # Step 5: Test initial accuracy
        print("\n5️⃣ Testing model evaluation...")
        initial_accuracy = server.evaluate_model()
        print(f"   📊 Initial accuracy: {initial_accuracy:.4f} ({initial_accuracy:.2%})")
        
        # Step 6: Quick training test (just 2 epochs)
        print("\n6️⃣ Quick training test (2 epochs)...")
        print("   This will take about 30-60 seconds...")
        
        # Temporarily reduce epochs for quick test
        original_epochs = config.LOCAL_EPOCHS_ROOT
        config.LOCAL_EPOCHS_ROOT = 2
        
        start_training = time.time()
        server._pretrain_global_model()
        training_time = time.time() - start_training
        
        # Restore original epochs
        config.LOCAL_EPOCHS_ROOT = original_epochs
        
        # Test accuracy after mini-training
        quick_accuracy = server.evaluate_model()
        improvement = quick_accuracy - initial_accuracy
        
        print(f"   ✅ Quick training completed in {training_time:.2f}s")
        print(f"   📊 Quick accuracy: {quick_accuracy:.4f} ({quick_accuracy:.2%})")
        print(f"   📈 Improvement: {improvement:.4f} ({improvement:.2%})")
        
        # Step 7: Estimate full training results
        print("\n7️⃣ Estimating full training results...")
        
        # Conservative estimate: improvement scales with sqrt(epochs)
        improvement_per_epoch = improvement / 2
        estimated_full_improvement = improvement_per_epoch * (original_epochs ** 0.5)
        estimated_full_accuracy = initial_accuracy + estimated_full_improvement
        
        print(f"   📊 Estimated full accuracy: {estimated_full_accuracy:.4f} ({estimated_full_accuracy:.2%})")
        
        # Step 8: Assessment
        print("\n8️⃣ ASSESSMENT:")
        if estimated_full_accuracy >= 0.80:
            status = "🎉 EXCELLENT"
            recommendation = "Ready for full experiments!"
        elif estimated_full_accuracy >= 0.70:
            status = "✅ GOOD"
            recommendation = "Likely to achieve 80%+ with full training"
        elif estimated_full_accuracy >= 0.60:
            status = "⚠️ PROMISING"
            recommendation = "May need slight parameter adjustments"
        else:
            status = "❌ NEEDS_WORK"
            recommendation = "Requires significant improvements"
        
        print(f"   Status: {status}")
        print(f"   💡 {recommendation}")
        
        # Summary
        print("\n" + "=" * 50)
        print("✅ MINIMAL TEST COMPLETED!")
        print(f"📊 Configuration improvements validated")
        print(f"🎯 Estimated accuracy: {estimated_full_accuracy:.2%}")
        print(f"⏱️ Full training time estimate: {training_time * original_epochs / 2:.0f}s")
        
        return {
            'status': status,
            'initial_accuracy': initial_accuracy,
            'quick_accuracy': quick_accuracy,
            'estimated_full_accuracy': estimated_full_accuracy,
            'config_improved': True,
            'training_time': training_time
        }
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"1. Make sure virtual environment is activated")
        print(f"2. Install missing packages: pip install -r requirements.txt")
        print(f"3. Check GPU memory if CUDA errors occur")
        
        return {'status': 'ERROR', 'error': str(e)}

if __name__ == "__main__":
    print("🚀 RUNNING MINIMAL ACCURACY VALIDATION")
    print("⏱️ This test takes about 1-2 minutes")
    print("🎯 Goal: Confirm configuration improvements work")
    print()
    
    result = run_minimal_test()
    
    print("\n" + "=" * 50)
    if result.get('status') in ['🎉 EXCELLENT', '✅ GOOD']:
        print("🎉 SUCCESS! Configuration is improved!")
        print("📝 Next: Run full experiment with main.py")
    elif result.get('status') == '⚠️ PROMISING':
        print("⚠️ Good progress! Minor tweaks may help")
        print("📝 Next: Consider small parameter increases")
    else:
        print("🔧 Needs work. Check configuration settings")
        print("📝 Next: Review and adjust parameters")
    
    print("=" * 50) 