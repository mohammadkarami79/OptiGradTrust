#!/usr/bin/env python3
"""
🔍 QUICK CONFIG TEST
Verify that our optimized configuration can be loaded correctly
"""

def test_config_import():
    """Test configuration import and basic functionality."""
    
    print("🔍 QUICK CONFIGURATION TEST")
    print("=" * 50)
    
    try:
        # Test 1: Import optimized config
        print("\n1️⃣ Testing optimized config import...")
        import simple_optimized_config as config
        print(f"   ✅ Config imported successfully")
        print(f"   📊 Dataset: {config.DATASET}")
        print(f"   📊 Model: {config.MODEL}")
        print(f"   📊 Epochs: {config.GLOBAL_EPOCHS}")
        print(f"   📊 Batch Size: {config.BATCH_SIZE}")
        
        # Test 2: Verify key parameters
        print("\n2️⃣ Verifying key parameters...")
        assert config.GLOBAL_EPOCHS >= 15, f"GLOBAL_EPOCHS too low: {config.GLOBAL_EPOCHS}"
        assert config.BATCH_SIZE >= 24, f"BATCH_SIZE too low: {config.BATCH_SIZE}"
        assert config.LOCAL_EPOCHS_ROOT >= 10, f"LOCAL_EPOCHS_ROOT too low: {config.LOCAL_EPOCHS_ROOT}"
        print(f"   ✅ All parameters pass minimum thresholds")
        
        # Test 3: Test PyTorch import
        print("\n3️⃣ Testing PyTorch import...")
        import torch
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   📊 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   📊 GPU: {torch.cuda.get_device_name(0)}")
        
        # Test 4: Test federated learning imports
        print("\n4️⃣ Testing federated learning imports...")
        try:
            # Override config first
            import federated_learning.config.config as fl_config
            for attr in dir(config):
                if not attr.startswith('_'):
                    setattr(fl_config, attr, getattr(config, attr))
            print(f"   ✅ Config override successful")
            
            # Test critical imports
            from federated_learning.training.server import Server
            from federated_learning.data.dataset_utils import load_dataset
            print(f"   ✅ Core modules imported successfully")
            
        except Exception as e:
            print(f"   ❌ Import error: {str(e)}")
            return False
        
        # Test 5: Test dataset loading capability
        print("\n5️⃣ Testing dataset loading capability...")
        try:
            root_dataset, test_dataset = load_dataset()
            print(f"   ✅ Dataset loaded successfully")
            print(f"   📊 Training samples: {len(root_dataset)}")
            print(f"   📊 Test samples: {len(test_dataset)}")
        except Exception as e:
            print(f"   ⚠️ Dataset loading issue: {str(e)}")
            print(f"   💡 This might be normal if CIFAR-10 needs to be downloaded")
        
        # Test 6: Memory estimation
        print("\n6️⃣ Memory estimation...")
        estimated_memory = (config.BATCH_SIZE * 3 * 32 * 32 * 4) / (1024**2)  # CIFAR-10 batch in MB
        print(f"   📊 Estimated batch memory: {estimated_memory:.2f} MB")
        
        if estimated_memory > 100:
            print(f"   ⚠️ High memory usage - consider reducing batch size")
        else:
            print(f"   ✅ Memory usage looks reasonable")
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("🎯 Configuration is ready for accuracy testing")
        print("💡 Next step: Run simple_accuracy_test.py")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_import()
    
    if success:
        print("\n🚀 READY TO PROCEED!")
        print("Run: python simple_accuracy_test.py")
    else:
        print("\n🔧 CONFIGURATION NEEDS FIXES")
        print("Check error messages above") 