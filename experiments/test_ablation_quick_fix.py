"""
Test سریع برای تأیید اصلاح Ablation Study v2
این تست فقط 3 rounds اجرا می‌کند تا سریع مشکل را identify کند.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
from experiments.ablation_study_v2 import run_enhanced_ablation_experiment

def set_seed(seed):
    """تنظیم seed برای reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    print("\n" + "="*80)
    print("🧪 QUICK FIX TEST - Ablation Study v2")
    print("="*80)
    
    # تنظیمات خیلی کوچک برای test سریع
    num_rounds = 3  # فقط 3 rounds
    attack_intensity = 30.0
    malicious_ratio = 0.4
    
    print(f"\n📝 Test Configuration:")
    print(f"  Rounds: {num_rounds}")
    print(f"  Attack Intensity: {attack_intensity}")
    print(f"  Malicious Ratio: {malicious_ratio}")
    print(f"  Dataset: MNIST")
    
    results = {}
    
    # =====================================================================
    # Test 1: Baseline (همه فعال)
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 1: Baseline (RL + Dual Attention)")
    print("="*80)
    
    set_seed(42)
    result_baseline = run_enhanced_ablation_experiment(
        config_name='mnist',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=False,
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=['scaling_attack'],
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=42
    )
    results['baseline'] = result_baseline
    
    # =====================================================================
    # Test 2: بدون RL (فقط Dual Attention)
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 2: Without RL (Only Dual Attention)")
    print("="*80)
    
    set_seed(42)
    result_no_rl = run_enhanced_ablation_experiment(
        config_name='mnist',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=True,  # ⚠️ RL disabled
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=['scaling_attack'],
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=42
    )
    results['without_rl'] = result_no_rl
    
    # =====================================================================
    # نتایج
    # =====================================================================
    print("\n" + "="*80)
    print("📊 QUICK FIX TEST RESULTS")
    print("="*80)
    
    for scenario, result in results.items():
        print(f"\n{scenario}:")
        print(f"  Final Accuracy: {result['final_accuracy']:.4f}")
        print(f"  Improvement: {result['improvement']:+.4f}")
        print(f"  Detection F1: {result['detection_f1']:.4f}")
    
    # تحلیل تفاوت
    baseline_acc = results['baseline']['final_accuracy']
    no_rl_acc = results['without_rl']['final_accuracy']
    diff = abs(baseline_acc - no_rl_acc)
    
    print("\n" + "="*80)
    print("📈 ANALYSIS")
    print("="*80)
    
    if diff > 0.001:
        print(f"✅ SUCCESS! تفاوت معنی‌دار یافت شد: {diff:.4f}")
        print(f"   Baseline: {baseline_acc:.4f}")
        print(f"   Without RL: {no_rl_acc:.4f}")
        print("\n✨ اصلاح موفق بود! می‌توانیم تست کامل را اجرا کنیم.")
    else:
        print(f"⚠️  هنوز تفاوت خیلی کم است: {diff:.4f}")
        print(f"   ممکن است نیاز به:")
        print(f"   - Rounds بیشتر (حداقل 10-15)")
        print(f"   - Attack قوی‌تر")
        print(f"   - بررسی logs برای مشکلات دیگر")
    
    print("\n" + "="*80)
    print(f"⏱  Test تمام شد!")
    print("="*80)

if __name__ == '__main__':
    main()

