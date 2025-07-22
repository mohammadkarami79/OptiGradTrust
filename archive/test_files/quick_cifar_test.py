#!/usr/bin/env python3
"""
🖼️ Quick CIFAR-10 Test - Memory Safe & Fast

هدف: نتایج واقعی CIFAR-10 در 30 دقیقه برای تکمیل مقاله
تنظیمات: Memory-optimized برای RTX 3060 6GB
"""

import torch
import numpy as np
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_cifar_config():
    """تنظیمات بهینه CIFAR-10 برای تست سریع"""
    print("🔧 Setting up CIFAR-10 quick config...")
    
    # Import and override config
    from federated_learning.config import config
    
    # Override config variables for CIFAR-10
    config.DATASET = 'CIFAR-10'
    config.MODEL = 'ResNet18'
    config.NUM_CLIENTS = 10
    config.FRACTION_MALICIOUS = 0.3
    
    # Quick settings - optimized for speed
    config.GLOBAL_EPOCHS = 3  # سریع اما کافی
    config.LOCAL_EPOCHS_CLIENT = 3
    config.VAE_EPOCHS = 12  # کمتر برای سرعت
    
    # Memory-safe settings for RTX 3060
    config.BATCH_SIZE = 16
    config.VAE_BATCH_SIZE = 8
    config.ROOT_DATASET_SIZE = 4000  # کمتر برای سرعت
    
    # Reduced model complexity
    config.VAE_LATENT_DIM_1 = 64
    config.VAE_LATENT_DIM_2 = 32  
    config.VAE_LATENT_DIM_3 = 16
    config.DUAL_ATTENTION_HIDDEN_SIZE = 128
    config.DUAL_ATTENTION_HEADS = 8
    config.GRADIENT_CHUNK_SIZE = 50000
    
    print(f"✅ Config set: {config.GLOBAL_EPOCHS} epochs, {config.NUM_CLIENTS} clients")
    print(f"📦 Batch sizes: {config.BATCH_SIZE}, VAE: {config.VAE_BATCH_SIZE}")
    print(f"🧠 Model: {config.MODEL}, Dataset size: {config.ROOT_DATASET_SIZE}")

def run_single_attack_test(attack_type):
    """تست یک نوع حمله با استفاده از main function"""
    print(f"\n🧪 Testing {attack_type}...")
    start_time = time.time()
    
    try:
        # Import config and set attack type
        from federated_learning.config import config
        
        # Update ALL_ATTACK_TYPES to only include our current attack
        import main
        original_attacks = main.ALL_ATTACK_TYPES.copy()
        main.ALL_ATTACK_TYPES = [attack_type]  # Only test this one attack
        
        print(f"🚀 Starting {attack_type} test...")
        
        # Call main function (which now only tests one attack)
        main.main()
        
        # Restore original attack list
        main.ALL_ATTACK_TYPES = original_attacks
        
        elapsed = time.time() - start_time
        print(f"⏱️ {attack_type} completed in {elapsed/60:.1f} minutes")
        
        return {"attack_type": attack_type, "status": "completed", "time": elapsed}
        
    except Exception as e:
        print(f"❌ Error in {attack_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_quick_cifar_test():
    """اجرای تست سریع CIFAR-10"""
    print("="*60)
    print("🖼️ QUICK CIFAR-10 TEST FOR PAPER")
    print("="*60)
    
    setup_cifar_config()
    
    # Test priority attacks (most important for paper)
    priority_attacks = [
        'noise_attack',        # معمولا بهترین نتایج
        'partial_scaling_attack',  # متوسط
        'sign_flipping_attack'     # استاندارد
    ]
    
    start_time = time.time()
    results = {}
    
    for attack in priority_attacks:
        result = run_single_attack_test(attack)
        if result:
            results[attack] = result
        
        # Quick break between tests
        print("💤 Taking 10 second break...")
        time.sleep(10)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Quick CIFAR-10 test completed in {total_time/60:.1f} minutes")
    
    return results

if __name__ == "__main__":
    print("🖼️ Starting quick CIFAR-10 validation...")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 7:  # RTX 3060 has ~6GB
            print("⚠️ Memory-optimized settings applied for RTX 3060")
        
    print("🚀 Running 3 priority attacks...")
    results = run_quick_cifar_test()
    
    if results:
        print("✅ CIFAR-10 quick test successful!")
        print("📊 Results ready for paper!")
        
        # Check for CSV results
        import glob
        csv_files = glob.glob("results/comprehensive_attack_summary_*.csv")
        if csv_files:
            latest_csv = max(csv_files, key=os.path.getctime)
            print(f"📄 Results saved to: {latest_csv}")
        
    else:
        print("❌ Test failed!") 