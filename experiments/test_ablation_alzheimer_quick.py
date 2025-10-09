"""
Test سریع با Alzheimer Dataset
این تست با dataset واقعی و 10 rounds اجرا می‌شود.
Alzheimer dataset سخت‌تر است و تفاوت‌ها واضح‌تری نشان می‌دهد.
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
    print("🧪 ALZHEIMER DATASET TEST - Ablation Study v2")
    print("="*80)
    
    # تنظیمات برای Alzheimer
    num_rounds = 10  # کافی برای دیدن تفاوت
    attack_intensity = 30.0
    malicious_ratio = 0.4
    
    print(f"\n📝 Test Configuration:")
    print(f"  Dataset: Alzheimer MRI")
    print(f"  Rounds: {num_rounds}")
    print(f"  Attack Intensity: {attack_intensity}")
    print(f"  Malicious Ratio: {malicious_ratio}")
    print(f"  Attack Type: scaling_attack")
    
    results = {}
    
    # =====================================================================
    # Test 1: Baseline (RL + Dual Attention)
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 1: Baseline (RL + Dual Attention)")
    print("  This is the full OptiGradTrust with RL")
    print("="*80)
    
    set_seed(42)
    result_baseline = run_enhanced_ablation_experiment(
        config_name='alzheimer',  # ← Alzheimer!
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
    
    print("\n✅ Baseline completed!")
    print(f"   Accuracy: {result_baseline['final_accuracy']:.4f}")
    print(f"   Detection F1: {result_baseline['detection_f1']:.4f}")
    
    # =====================================================================
    # Test 2: بدون RL (فقط Dual Attention)
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 2: Without RL (Only Dual Attention)")
    print("  This tests OptiGradTrust WITHOUT RL component")
    print("="*80)
    
    set_seed(42)
    result_no_rl = run_enhanced_ablation_experiment(
        config_name='alzheimer',  # ← Alzheimer!
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
    
    print("\n✅ Without RL completed!")
    print(f"   Accuracy: {result_no_rl['final_accuracy']:.4f}")
    print(f"   Detection F1: {result_no_rl['detection_f1']:.4f}")
    
    # =====================================================================
    # نتایج و تحلیل
    # =====================================================================
    print("\n" + "="*80)
    print("📊 ALZHEIMER TEST RESULTS")
    print("="*80)
    
    for scenario, result in results.items():
        print(f"\n{scenario}:")
        print(f"  Final Accuracy: {result['final_accuracy']:.4f}")
        print(f"  Improvement: {result['improvement']:+.4f}")
        print(f"  Detection Precision: {result['detection_precision']:.4f}")
        print(f"  Detection Recall: {result['detection_recall']:.4f}")
        print(f"  Detection F1: {result['detection_f1']:.4f}")
    
    # تحلیل تفاوت
    baseline_acc = results['baseline']['final_accuracy']
    no_rl_acc = results['without_rl']['final_accuracy']
    acc_diff = abs(baseline_acc - no_rl_acc)
    
    baseline_f1 = results['baseline']['detection_f1']
    no_rl_f1 = results['without_rl']['detection_f1']
    f1_diff = abs(baseline_f1 - no_rl_f1)
    
    print("\n" + "="*80)
    print("📈 DETAILED ANALYSIS")
    print("="*80)
    
    print(f"\n🎯 Accuracy Comparison:")
    print(f"  Baseline (with RL):    {baseline_acc:.4f}")
    print(f"  Without RL:            {no_rl_acc:.4f}")
    print(f"  Absolute Difference:   {acc_diff:.4f}")
    print(f"  Relative Difference:   {(acc_diff/baseline_acc*100):.2f}%")
    
    print(f"\n🎯 Detection F1 Comparison:")
    print(f"  Baseline (with RL):    {baseline_f1:.4f}")
    print(f"  Without RL:            {no_rl_f1:.4f}")
    print(f"  Absolute Difference:   {f1_diff:.4f}")
    if baseline_f1 > 0:
        print(f"  Relative Difference:   {(f1_diff/baseline_f1*100):.2f}%")
    
    # تفسیر نتایج
    print("\n" + "="*80)
    print("🔍 INTERPRETATION")
    print("="*80)
    
    if acc_diff > 0.01:
        print(f"\n✅ SUCCESS! تفاوت معنی‌دار در Accuracy یافت شد!")
        print(f"   Difference: {acc_diff:.4f} (>{0.01:.4f})")
        if baseline_acc > no_rl_acc:
            print(f"   → RL بهبود قابل توجهی ایجاد کرده است ✨")
        else:
            print(f"   → Without RL بهتر عمل کرده (جالب است!) 🤔")
    elif acc_diff > 0.005:
        print(f"\n⚠️  تفاوت کوچک اما قابل توجه در Accuracy")
        print(f"   Difference: {acc_diff:.4f}")
        print(f"   → ممکن است نیاز به rounds بیشتر باشد")
    else:
        print(f"\n❌ تفاوت در Accuracy خیلی کم است: {acc_diff:.4f}")
        print(f"   → نیاز به تحلیل بیشتر")
    
    if f1_diff > 0.05:
        print(f"\n✅ تفاوت معنی‌دار در Detection F1 یافت شد!")
        print(f"   Difference: {f1_diff:.4f} (>{0.05:.4f})")
        if baseline_f1 > no_rl_f1:
            print(f"   → RL در detection بهتر عمل می‌کند ✨")
        else:
            print(f"   → Without RL در detection بهتر است 🤔")
    
    # توصیه‌ها
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    if acc_diff > 0.01 or f1_diff > 0.05:
        print("\n✨ نتایج امیدوارکننده است!")
        print("\n📌 قدم‌های بعدی:")
        print("  1. اجرای تست با rounds بیشتر (30-50)")
        print("  2. تست با attack های متنوع")
        print("  3. اجرای کامل ablation study")
    else:
        print("\n⚠️  تفاوت هنوز کم است.")
        print("\n📌 پیشنهادات:")
        print("  1. افزایش تعداد rounds به 20-30")
        print("  2. افزایش attack intensity به 50")
        print("  3. استفاده از combined attacks")
        print("  4. بررسی دقیق logs برای مطمئن شدن از disable شدن RL")
    
    print("\n" + "="*80)
    print(f"⏱  Test تمام شد!")
    print(f"📁 نتایج ذخیره شده در: experiments/results/")
    print("="*80)

if __name__ == '__main__':
    main()

