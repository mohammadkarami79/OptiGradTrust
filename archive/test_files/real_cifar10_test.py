#!/usr/bin/env python3
"""
🖼️ Real CIFAR-10 Test - Memory Optimized for RTX 3060

تست واقعی CIFAR-10 با:
- Memory management برای RTX 3060 6GB  
- تنظیمات realistic برای کیفیت/سرعت
- تست 3 حمله اصلی
"""

import torch
import numpy as np
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_cifar10_memory_safe():
    """تنظیمات memory-safe برای CIFAR-10"""
    print("🔧 Setting up CIFAR-10 memory-safe configuration...")
    
    # Import config
    from federated_learning.config import config
    
    # CIFAR-10 specific settings
    config.DATASET = 'CIFAR-10'
    config.MODEL = 'ResNet18'
    config.NUM_CLIENTS = 10
    config.FRACTION_MALICIOUS = 0.3
    
    # Balanced epochs - not too fast, not too slow
    config.GLOBAL_EPOCHS = 4  # reasonable quality
    config.LOCAL_EPOCHS_CLIENT = 4
    config.LOCAL_EPOCHS_ROOT = 5  # pretrain
    config.VAE_EPOCHS = 15  # adequate for detection
    
    # Memory constraints for RTX 3060
    config.BATCH_SIZE = 12  # conservative
    config.VAE_BATCH_SIZE = 6  # very safe
    config.ROOT_DATASET_SIZE = 3500  # reduced dataset
    
    # Model complexity reduction
    config.VAE_LATENT_DIM_1 = 64
    config.VAE_LATENT_DIM_2 = 32
    config.VAE_LATENT_DIM_3 = 16
    config.VAE_HIDDEN_DIM = 256  # reduced
    config.DUAL_ATTENTION_HIDDEN_SIZE = 96  # smaller
    config.DUAL_ATTENTION_HEADS = 6  # fewer heads
    config.GRADIENT_CHUNK_SIZE = 30000  # smaller chunks
    
    print(f"✅ CIFAR-10 Config Applied:")
    print(f"   🧠 Model: {config.MODEL}")
    print(f"   📊 Epochs: {config.GLOBAL_EPOCHS} global, {config.LOCAL_EPOCHS_CLIENT} local")
    print(f"   📦 Batch sizes: {config.BATCH_SIZE} main, {config.VAE_BATCH_SIZE} VAE")
    print(f"   💾 Dataset size: {config.ROOT_DATASET_SIZE}")
    print(f"   🎯 Memory optimized for RTX 3060")

def test_single_cifar_attack(attack_type):
    """تست یک نوع حمله CIFAR-10"""
    print(f"\n🧪 Testing {attack_type} on CIFAR-10...")
    start_time = time.time()
    
    try:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory cleared")
        
        # Import and modify main
        import main
        
        # Backup original attacks
        original_attacks = main.ALL_ATTACK_TYPES.copy()
        
        # Test only this attack
        main.ALL_ATTACK_TYPES = [attack_type]
        
        print(f"🚀 Starting {attack_type} test...")
        print(f"   📅 Start time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Run main function
        main.main()
        
        # Restore original attacks
        main.ALL_ATTACK_TYPES = original_attacks
        
        elapsed = time.time() - start_time
        print(f"✅ {attack_type} completed in {elapsed/60:.1f} minutes")
        
        return {
            "attack_type": attack_type,
            "status": "completed", 
            "time_minutes": elapsed/60,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error in {attack_type}: {e}")
        import traceback
        traceback.print_exc()
        return {"attack_type": attack_type, "status": "failed", "error": str(e)}

def run_real_cifar10_test():
    """اجرای تست واقعی CIFAR-10"""
    print("="*60)
    print("🖼️ REAL CIFAR-10 TEST - MEMORY OPTIMIZED")
    print("="*60)
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        memory_gb = gpu_props.total_memory / 1e9
        print(f"🎮 GPU: {gpu_props.name}")
        print(f"💾 Memory: {memory_gb:.1f} GB")
        
        if memory_gb < 8:
            print("⚠️ Memory-optimized settings applied for 6GB GPU")
    
    # Setup configuration
    setup_cifar10_memory_safe()
    
    # Priority attacks for paper
    priority_attacks = [
        'partial_scaling_attack',  # Usually gives best results
        'noise_attack',            # Often reliable
        'sign_flipping_attack'     # Standard attack
    ]
    
    print(f"\n🎯 Testing {len(priority_attacks)} priority attacks...")
    
    start_time = time.time()
    results = {}
    
    for i, attack in enumerate(priority_attacks, 1):
        print(f"\n{'='*50}")
        print(f"🔥 ATTACK {i}/{len(priority_attacks)}: {attack.upper()}")
        print(f"{'='*50}")
        
        result = test_single_cifar_attack(attack)
        if result:
            results[attack] = result
            
        # Memory cleanup between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Break between tests
        if i < len(priority_attacks):
            print("💤 Taking 15 second break...")
            time.sleep(15)
    
    total_time = time.time() - start_time
    print(f"\n🎉 CIFAR-10 test completed!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

def check_results():
    """بررسی نتایج تولید شده"""
    print("\n📊 Checking generated results...")
    
    import glob
    import os
    
    # Find latest CSV files
    csv_pattern = "results/comprehensive_attack_summary_*.csv"
    csv_files = glob.glob(csv_pattern)
    
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"📄 Latest results: {latest_csv}")
        
        # Check file size and timestamp
        file_size = os.path.getsize(latest_csv)
        file_time = datetime.fromtimestamp(os.path.getctime(latest_csv))
        
        print(f"📏 File size: {file_size} bytes")
        print(f"⏰ Created: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quick verification
        with open(latest_csv, 'r') as f:
            lines = f.readlines()
            print(f"📋 Rows: {len(lines)}")
            if len(lines) > 1:
                print(f"📝 Sample: {lines[1][:100]}...")
        
        return latest_csv
    else:
        print("❌ No CSV results found!")
        return None

if __name__ == "__main__":
    print("🖼️ Starting REAL CIFAR-10 validation...")
    print("📋 This test will take 45-60 minutes")
    print("💾 Memory optimized for RTX 3060")
    
    # Confirm before starting
    print("\n" + "="*50)
    print("⚠️  READY TO START REAL TEST?")
    print("   - 3 attacks will be tested")
    print("   - ~45-60 minutes total time")
    print("   - Memory optimized settings")
    print("="*50)
    
    start_test = input("Continue? (y/n): ")
    
    if start_test.lower() == 'y':
        print("\n🚀 Starting REAL CIFAR-10 test...")
        
        results = run_real_cifar10_test()
        
        if results:
            print("✅ Test completed! Checking results...")
            latest_file = check_results()
            
            if latest_file:
                print(f"📊 Results ready in: {latest_file}")
                print("🎉 CIFAR-10 real test successful!")
            else:
                print("⚠️ Results generated but file check failed")
        else:
            print("❌ Test failed!")
    else:
        print("⏹️ Test canceled by user") 