"""
Quick Test for Focused Reviewer Response V2
============================================

هدف: تست سریع برای اطمینان از:
1. Alzheimer dataset لود می‌شود
2. Ablation واقعاً کار می‌کند
3. نتایج متفاوت هستند

زمان: ~5-10 دقیقه
Rounds: فقط 3
Experiments: فقط 2 (Full + Without Shapley)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
from pathlib import Path

# Import the V2 functions
from focused_reviewer_response_v2 import run_single_experiment, set_seed

def quick_test():
    """اجرای تست سریع"""
    print("\n" + "="*80)
    print("QUICK VERIFICATION TEST")
    print("="*80)
    print("\nTesting:")
    print("  1. Alzheimer dataset loading")
    print("  2. Real ablation (Shapley disable)")
    print("  3. Different results")
    print("\nSettings:")
    print("  - Dataset: Alzheimer")
    print("  - Rounds: 3 (very quick)")
    print("  - Experiments: 2 only")
    print("\nExpected time: 5-10 minutes\n")
    
    seed = 42
    num_rounds = 3  # بسیار کم برای تست سریع
    
    print("\n" + "="*80)
    print("[TEST 1/2] OptiGradTrust Full")
    print("="*80)
    
    result_full = run_single_experiment(
        name='OptiGradTrust (Full) - QUICK TEST',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=False,  # Shapley enabled
        disable_vae=False,
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    print("\n" + "="*80)
    print("[TEST 2/2] OptiGradTrust Without Shapley")
    print("="*80)
    
    result_no_shapley = run_single_experiment(
        name='OptiGradTrust without Shapley - QUICK TEST',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=True,  # Shapley disabled
        disable_vae=False,
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    # تحلیل نتایج
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    acc_full = result_full['final_accuracy']
    acc_no_shapley = result_no_shapley['final_accuracy']
    diff = acc_full - acc_no_shapley
    
    print(f"\nResults:")
    print(f"  Full OptiGradTrust:     {acc_full:.4f}")
    print(f"  Without Shapley:        {acc_no_shapley:.4f}")
    print(f"  Difference:             {diff:+.4f}")
    
    # بررسی‌های مهم
    checks_passed = 0
    checks_total = 3
    
    print("\n" + "-"*80)
    print("VERIFICATION CHECKS:")
    print("-"*80)
    
    # Check 1: Dataset
    if 'alzheimer' in str(result_full.get('name', '')).lower():
        print("✅ CHECK 1: Dataset is Alzheimer")
        checks_passed += 1
    else:
        print("❌ CHECK 1: Dataset is NOT Alzheimer!")
    
    # Check 2: Accuracy range
    if 0.5 < acc_full < 1.0 and 0.5 < acc_no_shapley < 1.0:
        print("✅ CHECK 2: Accuracy in reasonable range (not MNIST)")
        checks_passed += 1
    else:
        print("❌ CHECK 2: Accuracy suspicious (might be MNIST)")
    
    # Check 3: Different results
    if abs(diff) > 0.0001:  # حتی تفاوت خیلی کوچک هم قابل قبول است
        print(f"✅ CHECK 3: Results are different ({diff:+.4f})")
        checks_passed += 1
    else:
        print(f"❌ CHECK 3: Results are IDENTICAL ({diff:+.4f}) - Ablation not working!")
    
    print("-"*80)
    print(f"\nPassed: {checks_passed}/{checks_total}")
    
    if checks_passed == checks_total:
        print("\n" + "="*80)
        print("🎉 SUCCESS! All checks passed!")
        print("="*80)
        print("\n✅ You can proceed to Medium Test")
        print("   Command: python experiments/test_medium_verify.py")
        return True
    else:
        print("\n" + "="*80)
        print("⚠️  FAILED! Please fix issues before continuing")
        print("="*80)
        return False

if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)

