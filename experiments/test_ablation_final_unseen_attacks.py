"""
Final Aggressive Ablation Test with UNSEEN Attacks
===================================================

این تست برای نشان دادن قدرت RL در مقابل حملات جدید (unseen) است.

Strategy:
---------
1. Baseline: با RL - حملات unseen (partial_scaling, alternating, targeted)
2. Without RL: فقط Dual Attention - همان حملات unseen
3. RL باید بتواند adapt کند و بهتر عمل کند!

Key Insight (از کاربر):
-------------------------
Dual Attention روی حملات خاص train شده است.
RL می‌تواند روی حملات جدید یاد بگیرد و adapt کند.
→ این نقطه قوت RL است!
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
    print("🔥 FINAL AGGRESSIVE TEST - UNSEEN ATTACKS")
    print("="*80)
    
    print("\n💡 Strategy:")
    print("  - Use UNSEEN attacks that Dual Attention hasn't seen before")
    print("  - RL should adapt and perform better than fixed Dual Attention")
    print("  - This tests RL's adaptive capability!")
    
    # تنظیمات aggressive
    num_rounds = 50
    attack_intensity = 100.0  # خیلی شدید!
    malicious_ratio = 0.5  # 50% malicious
    
    # حملات UNSEEN - که Dual Attention روی آنها train نشده!
    unseen_attacks = [
        'partial_scaling_attack',   # فقط بخشی از gradient را scale می‌کند
        'alternating_attack',       # در هر round متفاوت عمل می‌کند
        'targeted_attack',          # روی کلاس خاص attack می‌کند
        'gradient_inversion_attack' # سعی در بازسازی data دارد
    ]
    
    print(f"\n📝 Test Configuration:")
    print(f"  Dataset: Alzheimer MRI")
    print(f"  Rounds: {num_rounds} (enough for RL to learn)")
    print(f"  Attack Intensity: {attack_intensity} (VERY AGGRESSIVE!)")
    print(f"  Malicious Ratio: {malicious_ratio} (50%!)")
    print(f"  Attack Types: {unseen_attacks}")
    print(f"  → These are UNSEEN attacks!")
    
    results = {}
    
    # =====================================================================
    # Test 1: Baseline (RL + Dual Attention) با UNSEEN attacks
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 1: Baseline (RL + Dual Attention) - UNSEEN Attacks")
    print("  RL should LEARN to handle these new attacks!")
    print("="*80)
    
    set_seed(42)
    result_baseline = run_enhanced_ablation_experiment(
        config_name='alzheimer',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=False,  # RL ENABLED
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=unseen_attacks,  # UNSEEN!
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=42
    )
    results['baseline_rl_unseen'] = result_baseline
    
    print("\n✅ Baseline with RL completed!")
    print(f"   Final Accuracy: {result_baseline['final_accuracy']:.4f}")
    print(f"   Detection F1: {result_baseline['detection_f1']:.4f}")
    
    # =====================================================================
    # Test 2: Without RL (فقط Dual Attention) با همان UNSEEN attacks
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 2: Without RL (Only Dual Attention) - UNSEEN Attacks")
    print("  Dual Attention is FIXED - cannot adapt to new attacks!")
    print("="*80)
    
    set_seed(42)
    result_no_rl = run_enhanced_ablation_experiment(
        config_name='alzheimer',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=True,  # RL DISABLED
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=unseen_attacks,  # همان UNSEEN attacks
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=42
    )
    results['no_rl_unseen'] = result_no_rl
    
    print("\n✅ Without RL completed!")
    print(f"   Final Accuracy: {result_no_rl['final_accuracy']:.4f}")
    print(f"   Detection F1: {result_no_rl['detection_f1']:.4f}")
    
    # =====================================================================
    # Test 3 (BONUS): Baseline با KNOWN attacks (برای مقایسه)
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 3 (BONUS): Baseline with KNOWN Attacks")
    print("  For comparison - both should work well on known attacks")
    print("="*80)
    
    known_attacks = ['scaling_attack', 'sign_flipping']
    
    set_seed(42)
    result_baseline_known = run_enhanced_ablation_experiment(
        config_name='alzheimer',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=False,
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=known_attacks,  # KNOWN attacks
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=42
    )
    results['baseline_rl_known'] = result_baseline_known
    
    print("\n✅ Baseline with known attacks completed!")
    print(f"   Final Accuracy: {result_baseline_known['final_accuracy']:.4f}")
    print(f"   Detection F1: {result_baseline_known['detection_f1']:.4f}")
    
    # =====================================================================
    # نتایج و تحلیل جامع
    # =====================================================================
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS")
    print("="*80)
    
    print("\n### UNSEEN ATTACKS (New/Unknown) ###")
    print(f"\nBaseline (with RL):")
    print(f"  Final Accuracy: {results['baseline_rl_unseen']['final_accuracy']:.4f}")
    print(f"  Detection Precision: {results['baseline_rl_unseen']['detection_precision']:.4f}")
    print(f"  Detection Recall: {results['baseline_rl_unseen']['detection_recall']:.4f}")
    print(f"  Detection F1: {results['baseline_rl_unseen']['detection_f1']:.4f}")
    
    print(f"\nWithout RL (Dual Attention only):")
    print(f"  Final Accuracy: {results['no_rl_unseen']['final_accuracy']:.4f}")
    print(f"  Detection Precision: {results['no_rl_unseen']['detection_precision']:.4f}")
    print(f"  Detection Recall: {results['no_rl_unseen']['detection_recall']:.4f}")
    print(f"  Detection F1: {results['no_rl_unseen']['detection_f1']:.4f}")
    
    print("\n### KNOWN ATTACKS (For Reference) ###")
    print(f"\nBaseline (with RL):")
    print(f"  Final Accuracy: {results['baseline_rl_known']['final_accuracy']:.4f}")
    print(f"  Detection F1: {results['baseline_rl_known']['detection_f1']:.4f}")
    
    # تحلیل دقیق تفاوت‌ها
    acc_diff_unseen = results['baseline_rl_unseen']['final_accuracy'] - results['no_rl_unseen']['final_accuracy']
    f1_diff_unseen = results['baseline_rl_unseen']['detection_f1'] - results['no_rl_unseen']['detection_f1']
    
    print("\n" + "="*80)
    print("📈 DETAILED ANALYSIS - UNSEEN ATTACKS")
    print("="*80)
    
    print(f"\n🎯 Accuracy Comparison (UNSEEN attacks):")
    print(f"  Baseline (RL):      {results['baseline_rl_unseen']['final_accuracy']:.4f}")
    print(f"  Without RL:         {results['no_rl_unseen']['final_accuracy']:.4f}")
    print(f"  Difference:         {acc_diff_unseen:+.4f}")
    if acc_diff_unseen > 0:
        print(f"  → RL is BETTER by {acc_diff_unseen:.4f} ✅")
    elif acc_diff_unseen < 0:
        print(f"  → Dual Attention is better by {abs(acc_diff_unseen):.4f} ⚠️")
    else:
        print(f"  → Same performance")
    
    print(f"\n🎯 Detection F1 Comparison (UNSEEN attacks):")
    print(f"  Baseline (RL):      {results['baseline_rl_unseen']['detection_f1']:.4f}")
    print(f"  Without RL:         {results['no_rl_unseen']['detection_f1']:.4f}")
    print(f"  Difference:         {f1_diff_unseen:+.4f}")
    if f1_diff_unseen > 0:
        print(f"  → RL is BETTER by {f1_diff_unseen:.4f} ✅")
    elif f1_diff_unseen < 0:
        print(f"  → Dual Attention is better by {abs(f1_diff_unseen):.4f} ⚠️")
    else:
        print(f"  → Same performance")
    
    # تفسیر نتایج
    print("\n" + "="*80)
    print("🔍 INTERPRETATION")
    print("="*80)
    
    if acc_diff_unseen > 0.01 or f1_diff_unseen > 0.05:
        print("\n✅ SUCCESS! RL shows CLEAR advantage on UNSEEN attacks!")
        print(f"   Accuracy improvement: {acc_diff_unseen:.4f}")
        print(f"   F1 improvement: {f1_diff_unseen:.4f}")
        print("\n💡 Key Finding:")
        print("   → RL's adaptive capability allows it to handle NEW attacks")
        print("   → Dual Attention is fixed and struggles with unseen patterns")
        print("   → This validates RL's contribution to the framework!")
        
    elif acc_diff_unseen > 0.005 or f1_diff_unseen > 0.02:
        print("\n⚠️  RL shows MODERATE advantage on UNSEEN attacks")
        print(f"   Accuracy improvement: {acc_diff_unseen:.4f}")
        print(f"   F1 improvement: {f1_diff_unseen:.4f}")
        print("\n💡 Key Finding:")
        print("   → RL provides some adaptation to new attacks")
        print("   → But the advantage is not as strong as expected")
        print("   → May need stronger attacks or more rounds")
        
    else:
        print("\n❌ SURPRISING: RL does NOT show advantage on UNSEEN attacks")
        print(f"   Accuracy difference: {acc_diff_unseen:.4f}")
        print(f"   F1 difference: {f1_diff_unseen:.4f}")
        print("\n🤔 Possible reasons:")
        print("   1. Dual Attention generalizes well to unseen attacks")
        print("   2. RL needs more rounds to learn (try 100+)")
        print("   3. Attacks may not be diverse enough")
        print("   4. Hybrid blending reduces RL's impact")
    
    # مقایسه UNSEEN vs KNOWN
    acc_diff_known_vs_unseen = results['baseline_rl_known']['final_accuracy'] - results['baseline_rl_unseen']['final_accuracy']
    
    print("\n" + "="*80)
    print("📊 KNOWN vs UNSEEN Attacks (Baseline with RL)")
    print("="*80)
    
    print(f"\nKnown Attacks Accuracy:   {results['baseline_rl_known']['final_accuracy']:.4f}")
    print(f"Unseen Attacks Accuracy:  {results['baseline_rl_unseen']['final_accuracy']:.4f}")
    print(f"Difference:               {acc_diff_known_vs_unseen:+.4f}")
    
    if acc_diff_known_vs_unseen > 0.02:
        print("\n→ Unseen attacks are SIGNIFICANTLY harder (as expected) ✅")
    elif acc_diff_known_vs_unseen > 0:
        print("\n→ Unseen attacks are slightly harder")
    else:
        print("\n→ Unseen attacks have similar difficulty")
    
    # توصیه‌های نهایی
    print("\n" + "="*80)
    print("💡 FINAL RECOMMENDATIONS")
    print("="*80)
    
    if acc_diff_unseen > 0.01 or f1_diff_unseen > 0.05:
        print("\n✨ نتایج عالی است!")
        print("\n📝 For the paper:")
        print("  1. Highlight RL's adaptive capability on unseen attacks")
        print("  2. Show comparison: RL vs Dual Attention on new threats")
        print("  3. Emphasize: 'RL provides robustness to novel attack patterns'")
        print("\n📊 Table suggestion:")
        print("  | Method | Known Attacks | Unseen Attacks | Δ |")
        print("  |--------|--------------|----------------|---|")
        print(f"  | RL     | {results['baseline_rl_known']['final_accuracy']:.4f}        | {results['baseline_rl_unseen']['final_accuracy']:.4f}          | {acc_diff_known_vs_unseen:+.4f} |")
        print(f"  | No RL  | N/A          | {results['no_rl_unseen']['final_accuracy']:.4f}          | - |")
        
    else:
        print("\n⚠️  نتایج انتظار را برآورده نکرد")
        print("\n📌 Next steps:")
        print("  1. Try even MORE rounds (100-150)")
        print("  2. Use COMBINED unseen attacks")
        print("  3. Increase malicious ratio to 0.6-0.7")
        print("  4. Try with CIFAR-10 dataset")
        print("\n📝 For the paper (honest approach):")
        print("  - Report that RL provides marginal benefits")
        print("  - Highlight Dual Attention's strong generalization")
        print("  - Discuss: 'Pre-designed features capture patterns well'")
    
    print("\n" + "="*80)
    print(f"⏱  Final Aggressive Test completed!")
    print(f"📁 Results saved in: experiments/results/")
    print("="*80)
    
    # ذخیره نتایج برای reference
    import json
    output_file = f"experiments/results/final_unseen_attacks_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'rounds': num_rounds,
                'attack_intensity': attack_intensity,
                'malicious_ratio': malicious_ratio,
                'unseen_attacks': unseen_attacks,
                'known_attacks': known_attacks
            },
            'results': {
                'baseline_rl_unseen': {
                    'accuracy': float(results['baseline_rl_unseen']['final_accuracy']),
                    'f1': float(results['baseline_rl_unseen']['detection_f1']),
                    'precision': float(results['baseline_rl_unseen']['detection_precision']),
                    'recall': float(results['baseline_rl_unseen']['detection_recall'])
                },
                'no_rl_unseen': {
                    'accuracy': float(results['no_rl_unseen']['final_accuracy']),
                    'f1': float(results['no_rl_unseen']['detection_f1']),
                    'precision': float(results['no_rl_unseen']['detection_precision']),
                    'recall': float(results['no_rl_unseen']['detection_recall'])
                },
                'baseline_rl_known': {
                    'accuracy': float(results['baseline_rl_known']['final_accuracy']),
                    'f1': float(results['baseline_rl_known']['detection_f1'])
                }
            },
            'analysis': {
                'acc_diff_unseen': float(acc_diff_unseen),
                'f1_diff_unseen': float(f1_diff_unseen),
                'acc_diff_known_vs_unseen': float(acc_diff_known_vs_unseen)
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == '__main__':
    main()

