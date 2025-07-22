#!/usr/bin/env python3
"""
🚀 Rapid Test Validation - بهینه‌سازی سریع برای مقاله

هدف: نتایج بهتر برای MNIST + CIFAR-10 در کمترین زمان
روش: epochs متوسط، تنظیمات بهینه، 2 dataset
"""

import torch
import numpy as np
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mnist_optimized():
    """تست بهینه MNIST - epochs متوسط"""
    print("🧠 Testing MNIST with optimized config...")
    
    # Import with optimized settings
    from federated_learning.config.config import *
    
    # MNIST Optimized Config
    globals()['DATASET'] = 'MNIST'
    globals()['MODEL'] = 'CNN'
    globals()['NUM_CLIENTS'] = 10
    globals()['FRACTION_MALICIOUS'] = 0.3
    globals()['GLOBAL_EPOCHS'] = 5  # متوسط - نه خیلی کم، نه خیلی زیاد
    globals()['LOCAL_EPOCHS_CLIENT'] = 5
    globals()['VAE_EPOCHS'] = 20  # کافی برای detection
    globals()['BATCH_SIZE'] = 64
    
    from federated_learning.training.server import main
    
    print(f"🎯 MNIST Config: {GLOBAL_EPOCHS} epochs, {NUM_CLIENTS} clients")
    results = main()
    return results

def test_cifar_optimized():
    """تست بهینه CIFAR-10 - memory conscious"""
    print("🖼️ Testing CIFAR-10 with memory-optimized config...")
    
    from federated_learning.config.config import *
    
    # CIFAR-10 Memory-Optimized Config
    globals()['DATASET'] = 'CIFAR-10'
    globals()['MODEL'] = 'ResNet18'
    globals()['NUM_CLIENTS'] = 10
    globals()['FRACTION_MALICIOUS'] = 0.3
    globals()['GLOBAL_EPOCHS'] = 4  # سریع اما کافی
    globals()['LOCAL_EPOCHS_CLIENT'] = 4
    globals()['VAE_EPOCHS'] = 15
    globals()['BATCH_SIZE'] = 16  # memory safe
    globals()['VAE_BATCH_SIZE'] = 8
    
    from federated_learning.training.server import main
    
    print(f"🎯 CIFAR Config: {GLOBAL_EPOCHS} epochs, {NUM_CLIENTS} clients")
    results = main()
    return results

def run_rapid_validation():
    """اجرای تست سریع"""
    print("="*60)
    print("🚀 RAPID VALIDATION FOR PAPER RESULTS")
    print("="*60)
    
    start_time = time.time()
    results = {}
    
    try:
        # Test 1: MNIST
        print("\n📊 TEST 1: MNIST + CNN")
        print("-" * 40)
        mnist_results = test_mnist_optimized()
        results['MNIST'] = mnist_results
        
        # Test 2: CIFAR-10
        print("\n📊 TEST 2: CIFAR-10 + ResNet18")
        print("-" * 40)
        cifar_results = test_cifar_optimized()
        results['CIFAR-10'] = cifar_results
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        return None
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total validation time: {total_time/60:.1f} minutes")
    
    return results

if __name__ == "__main__":
    print("🧪 Starting rapid validation for paper...")
    
    # Run only if needed
    choice = input("Run full validation? (y/n): ")
    if choice.lower() == 'y':
        results = run_rapid_validation()
        if results:
            print("✅ Rapid validation completed!")
        else:
            print("❌ Validation failed!")
    else:
        print("⏭️ Skipping full validation. Using existing results...")
        
        # Instead, let's organize existing results
        print("\n📁 Organizing existing results for paper...") 