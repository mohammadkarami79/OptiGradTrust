"""
Medium Test for Focused Reviewer Response V2
=============================================

هدف: تست متوسط برای اطمینان از:
1. همه ablation ها کار می‌کنند
2. FedAvg baseline تفاوت قابل توجه دارد
3. Multiple attacks کار می‌کنند

زمان: ~10-15 دقیقه
Rounds: 10
Experiments: 5 (Full + 3 ablations + 1 attack)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

# Import the V2 functions
from focused_reviewer_response_v2 import run_single_experiment, set_seed

def medium_test():
    """اجرای تست متوسط"""
    print("\n" + "="*80)
    print("MEDIUM VERIFICATION TEST")
    print("="*80)
    print("\nTesting:")
    print("  1. All ablation components")
    print("  2. FedAvg baseline comparison")
    print("  3. Different attack types")
    print("\nSettings:")
    print("  - Dataset: Alzheimer")
    print("  - Rounds: 10")
    print("  - Experiments: 5")
    print("\nExpected time: 10-15 minutes\n")
    
    seed = 42
    num_rounds = 10
    results = {}
    
    # 1. Full OptiGradTrust
    print("\n[1/5] OptiGradTrust Full...")
    results['full'] = run_single_experiment(
        name='OptiGradTrust (Full)',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=False,
        disable_vae=False,
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    # 2. Without Shapley
    print("\n[2/5] Without Shapley...")
    results['no_shapley'] = run_single_experiment(
        name='Without Shapley',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=True,
        disable_vae=False,
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    # 3. Without VAE
    print("\n[3/5] Without VAE...")
    results['no_vae'] = run_single_experiment(
        name='Without VAE',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=False,
        disable_vae=True,
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    # 4. FedAvg Baseline
    print("\n[4/5] FedAvg Baseline...")
    results['fedavg'] = run_single_experiment(
        name='FedAvg Baseline',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=True,
        disable_vae=True,
        aggregation_method='fedavg',
        seed=seed
    )
    
    # 5. Different attack
    print("\n[5/5] Label Flipping Attack...")
    results['label_flip'] = run_single_experiment(
        name='Label Flipping',
        attack_type='label_flipping',
        num_rounds=num_rounds,
        disable_shapley=False,
        disable_vae=False,
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    # تحلیل نتایج
    print("\n" + "="*80)
    print("MEDIUM TEST RESULTS")
    print("="*80)
    
    print(f"\n{'Configuration':<30} {'Accuracy':<12} {'vs Full':<12}")
    print("-" * 54)
    
    full_acc = results['full']['final_accuracy']
    
    for key, label in [
        ('full', 'Full OptiGradTrust'),
        ('no_shapley', 'Without Shapley'),
        ('no_vae', 'Without VAE'),
        ('fedavg', 'FedAvg Baseline'),
        ('label_flip', 'Label Flipping Attack')
    ]:
        acc = results[key]['final_accuracy']
        diff = acc - full_acc
        marker = " <- BASELINE" if key == 'full' else ""
        print(f"{label:<30} {acc:<12.4f} {diff:>+11.4f}{marker}")
    
    # Verification checks
    print("\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)
    
    checks_passed = 0
    checks_total = 5
    
    # Check 1: Shapley impact
    shapley_diff = full_acc - results['no_shapley']['final_accuracy']
    if abs(shapley_diff) > 0.001:
        print(f"✅ CHECK 1: Shapley makes a difference ({shapley_diff:+.4f})")
        checks_passed += 1
    else:
        print(f"❌ CHECK 1: Shapley has NO effect ({shapley_diff:+.4f})")
    
    # Check 2: VAE impact
    vae_diff = full_acc - results['no_vae']['final_accuracy']
    if abs(vae_diff) > 0.001:
        print(f"✅ CHECK 2: VAE makes a difference ({vae_diff:+.4f})")
        checks_passed += 1
    else:
        print(f"❌ CHECK 2: VAE has NO effect ({vae_diff:+.4f})")
    
    # Check 3: FedAvg baseline
    baseline_diff = full_acc - results['fedavg']['final_accuracy']
    if baseline_diff > 0.01:  # حداقل 1% بهتر
        print(f"✅ CHECK 3: OptiGradTrust beats FedAvg ({baseline_diff:+.4f})")
        checks_passed += 1
    else:
        print(f"❌ CHECK 3: OptiGradTrust NOT better than FedAvg ({baseline_diff:+.4f})")
    
    # Check 4: Accuracy range
    if 0.6 < full_acc < 0.99:
        print(f"✅ CHECK 4: Accuracy in Alzheimer range ({full_acc:.4f})")
        checks_passed += 1
    else:
        print(f"❌ CHECK 4: Accuracy suspicious ({full_acc:.4f})")
    
    # Check 5: Different attack works
    attack_diff = abs(results['full']['final_accuracy'] - results['label_flip']['final_accuracy'])
    if attack_diff > 0.001:
        print(f"✅ CHECK 5: Different attacks produce different results ({attack_diff:.4f})")
        checks_passed += 1
    else:
        print(f"❌ CHECK 5: Attacks produce SAME results ({attack_diff:.4f})")
    
    print("-"*80)
    print(f"\nPassed: {checks_passed}/{checks_total}")
    
    if checks_passed >= 4:  # حداقل 4 از 5
        print("\n" + "="*80)
        print("🎉 SUCCESS! Medium test passed!")
        print("="*80)
        print("\n✅ You can proceed to Full Experiment")
        print("   Command: python experiments/focused_reviewer_response_v2.py")
        
        print("\n📊 Expected results for full run (50 rounds):")
        print(f"   - OptiGradTrust:    ~{full_acc + 0.05:.2f} (will improve)")
        print(f"   - Without Shapley:  ~{full_acc + 0.03:.2f}")
        print(f"   - Without VAE:      ~{full_acc + 0.02:.2f}")
        print(f"   - FedAvg Baseline:  ~{results['fedavg']['final_accuracy']:.2f}")
        
        return True
    else:
        print("\n" + "="*80)
        print("⚠️  FAILED! Fix issues before full run")
        print("="*80)
        return False

if __name__ == '__main__':
    success = medium_test()
    sys.exit(0 if success else 1)

