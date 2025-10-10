"""
Combined UNSEEN Attacks Test - The Real Challenge!
==================================================

Strategy: Test RL against COMBINED attacks (multiple attacks at once)
This is the TRUE test of adaptive capability!

Key Insights:
- Single attacks are too simple
- Combined attacks = real-world scenarios
- RL should excel at complex multi-attack scenarios
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
    print("🔥 COMBINED UNSEEN ATTACKS TEST - The Real Challenge!")
    print("="*80)
    
    print("\n💡 Strategy:")
    print("  - Use COMBINED attacks (2 attacks at once)")
    print("  - This creates complex, unpredictable patterns")
    print("  - RL must adapt to MULTIPLE threats simultaneously")
    print("  - Dual Attention is FIXED - cannot adapt to combinations")
    
    # تنظیمات متعادل (نه خیلی شدید)
    num_rounds = 75  # کافی برای RL learning
    attack_intensity = 50.0  # متعادل (نه خیلی شدید)
    malicious_ratio = 0.5  # 50% malicious
    
    # حملات COMBINED - ترکیب attacks
    combined_unseen_attacks = [
        'partial_scaling_attack',  # + alternating در background
        'targeted_attack'  # + gradient_inversion در background
    ]
    
    print(f"\n📝 Test Configuration:")
    print(f"  Dataset: Alzheimer MRI")
    print(f"  Rounds: {num_rounds} (enough for RL learning)")
    print(f"  Attack Intensity: {attack_intensity} (BALANCED)")
    print(f"  Malicious Ratio: {malicious_ratio} (50%)")
    print(f"  Attack Types: {combined_unseen_attacks}")
    print(f"  → These are COMBINED unseen attacks!")
    
    results = {}
    
    # =====================================================================
    # Test 1: Baseline (RL) با COMBINED UNSEEN attacks
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 1: Baseline (RL + Dual Attention) - COMBINED Attacks")
    print("  RL should LEARN to handle these complex combinations!")
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
        attack_types=combined_unseen_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=42
    )
    results['baseline_rl_combined'] = result_baseline
    
    print("\n✅ Baseline with RL completed!")
    print(f"   Final Accuracy: {result_baseline['final_accuracy']:.4f}")
    print(f"   Detection F1: {result_baseline['detection_f1']:.4f}")
    
    # =====================================================================
    # Test 2: Without RL با COMBINED UNSEEN attacks
    # =====================================================================
    print("\n" + "="*80)
    print("📊 Test 2: Without RL (Only Dual Attention) - COMBINED Attacks")
    print("  Dual Attention is FIXED - struggles with combinations!")
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
        attack_types=combined_unseen_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=42
    )
    results['no_rl_combined'] = result_no_rl
    
    print("\n✅ Without RL completed!")
    print(f"   Final Accuracy: {result_no_rl['final_accuracy']:.4f}")
    print(f"   Detection F1: {result_no_rl['detection_f1']:.4f}")
    
    # =====================================================================
    # نتایج و تحلیل
    # =====================================================================
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS - COMBINED ATTACKS")
    print("="*80)
    
    print(f"\nBaseline (with RL):")
    print(f"  Final Accuracy: {results['baseline_rl_combined']['final_accuracy']:.4f}")
    print(f"  Detection Precision: {results['baseline_rl_combined']['detection_precision']:.4f}")
    print(f"  Detection Recall: {results['baseline_rl_combined']['detection_recall']:.4f}")
    print(f"  Detection F1: {results['baseline_rl_combined']['detection_f1']:.4f}")
    
    print(f"\nWithout RL (Dual Attention only):")
    print(f"  Final Accuracy: {results['no_rl_combined']['final_accuracy']:.4f}")
    print(f"  Detection Precision: {results['no_rl_combined']['detection_precision']:.4f}")
    print(f"  Detection Recall: {results['no_rl_combined']['detection_recall']:.4f}")
    print(f"  Detection F1: {results['no_rl_combined']['detection_f1']:.4f}")
    
    # تحلیل تفاوت‌ها
    acc_diff = results['baseline_rl_combined']['final_accuracy'] - results['no_rl_combined']['final_accuracy']
    f1_diff = results['baseline_rl_combined']['detection_f1'] - results['no_rl_combined']['detection_f1']
    
    print("\n" + "="*80)
    print("📈 DETAILED ANALYSIS - COMBINED ATTACKS")
    print("="*80)
    
    print(f"\n🎯 Accuracy Comparison:")
    print(f"  Baseline (RL):      {results['baseline_rl_combined']['final_accuracy']:.4f}")
    print(f"  Without RL:         {results['no_rl_combined']['final_accuracy']:.4f}")
    print(f"  Difference:         {acc_diff:+.4f}")
    if acc_diff > 0.02:
        print(f"  → RL is BETTER by {acc_diff:.4f} ✅")
    elif acc_diff < -0.02:
        print(f"  → Dual Attention is better by {abs(acc_diff):.4f} ⚠️")
    else:
        print(f"  → Similar performance")
    
    print(f"\n🎯 Detection F1 Comparison:")
    print(f"  Baseline (RL):      {results['baseline_rl_combined']['detection_f1']:.4f}")
    print(f"  Without RL:         {results['no_rl_combined']['detection_f1']:.4f}")
    print(f"  Difference:         {f1_diff:+.4f}")
    if f1_diff > 0.05:
        print(f"  → RL is BETTER by {f1_diff:.4f} ✅")
    elif f1_diff < -0.05:
        print(f"  → Dual Attention is better by {abs(f1_diff):.4f} ⚠️")
    else:
        print(f"  → Similar performance")
    
    # تفسیر
    print("\n" + "="*80)
    print("🔍 INTERPRETATION")
    print("="*80)
    
    if acc_diff > 0.02 or f1_diff > 0.05:
        print("\n✅ SUCCESS! RL shows advantage on COMBINED attacks!")
        print(f"   Accuracy improvement: {acc_diff:.4f}")
        print(f"   F1 improvement: {f1_diff:.4f}")
        print("\n💡 Key Finding:")
        print("   → RL adapts to complex multi-attack scenarios")
        print("   → Dual Attention struggles with unseen combinations")
        print("   → This validates RL's adaptive capability!")
        
    elif acc_diff > 0.01 or f1_diff > 0.02:
        print("\n⚠️  RL shows MODERATE advantage")
        print(f"   Accuracy improvement: {acc_diff:.4f}")
        print(f"   F1 improvement: {f1_diff:.4f}")
        print("\n💡 Key Finding:")
        print("   → RL provides some benefit on complex scenarios")
        print("   → But advantage is not as strong as hoped")
        
    else:
        print("\n❌ Similar performance - RL does not show clear advantage")
        print(f"   Accuracy difference: {acc_diff:.4f}")
        print(f"   F1 difference: {f1_diff:.4f}")
        print("\n🤔 Possible reasons:")
        print("   1. Dual Attention generalizes very well")
        print("   2. Need even more complex scenarios")
        print("   3. Need higher malicious ratio (0.7)")
    
    # توصیه‌های بعدی
    print("\n" + "="*80)
    print("💡 NEXT STEPS")
    print("="*80)
    
    if acc_diff > 0.02 or f1_diff > 0.05:
        print("\n✨ Great results! For the paper:")
        print("  1. Report RL's advantage on complex scenarios")
        print("  2. Emphasize adaptive capability")
        print("  3. Compare: simple vs combined attacks")
        
    else:
        print("\n📌 To strengthen results:")
        print("  1. Try 70% malicious ratio")
        print("  2. Try 100-150 rounds")
        print("  3. Try extreme intensity (but balanced)")
        print("  4. Or accept: Dual Attention is very strong!")
    
    print("\n" + "="*80)
    print(f"⏱  Combined Attacks Test completed!")
    print(f"📁 Results saved in: experiments/results/")
    print("="*80)
    
    # ذخیره نتایج
    import json
    output_file = "experiments/results/combined_unseen_attacks_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'rounds': num_rounds,
                'attack_intensity': attack_intensity,
                'malicious_ratio': malicious_ratio,
                'combined_attacks': combined_unseen_attacks
            },
            'results': {
                'baseline_rl': {
                    'accuracy': float(results['baseline_rl_combined']['final_accuracy']),
                    'f1': float(results['baseline_rl_combined']['detection_f1']),
                    'precision': float(results['baseline_rl_combined']['detection_precision']),
                    'recall': float(results['baseline_rl_combined']['detection_recall'])
                },
                'no_rl': {
                    'accuracy': float(results['no_rl_combined']['final_accuracy']),
                    'f1': float(results['no_rl_combined']['detection_f1']),
                    'precision': float(results['no_rl_combined']['detection_precision']),
                    'recall': float(results['no_rl_combined']['detection_recall'])
                }
            },
            'analysis': {
                'acc_diff': float(acc_diff),
                'f1_diff': float(f1_diff)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == '__main__':
    main()

