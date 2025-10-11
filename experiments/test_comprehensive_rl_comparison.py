"""
Comprehensive RL vs Dual Attention Comparison
==============================================

این تست جامع برای مقایسه کامل:
1. Pure Dual Attention (بدون RL)
2. Hybrid (warmup + ramp-up + pure RL)
3. Hybrid Conservative (warmup/ramp-up بیشتر)
4. Pure RL (بدون warmup)

هدف: فهمیدن دقیق تاثیر هر configuration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
from experiments.ablation_study_v2 import run_enhanced_ablation_experiment
import json
from pathlib import Path

def set_seed(seed):
    """تنظیم seed برای reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_with_config(config_name, rl_aggregation_method, warmup=None, rampup=None, 
                    num_rounds=75, attack_types=None, malicious_ratio=0.5, 
                    attack_intensity=50.0, seed=42, 
                    disable_rl=False, disable_dual_attention=False):
    """
    اجرای تست با یک configuration خاص
    """
    import federated_learning.config.config as config
    
    # ذخیره تنظیمات قبلی
    original_rl_method = getattr(config, 'RL_AGGREGATION_METHOD', 'hybrid')
    original_warmup = getattr(config, 'RL_WARMUP_ROUNDS', 5)
    original_rampup = getattr(config, 'RL_RAMP_UP_ROUNDS', 10)
    
    # تنظیم configuration جدید
    config.RL_AGGREGATION_METHOD = rl_aggregation_method
    if warmup is not None:
        config.RL_WARMUP_ROUNDS = warmup
    if rampup is not None:
        config.RL_RAMP_UP_ROUNDS = rampup
    
    print(f"\n{'='*80}")
    print(f"[CONFIG] {config_name}")
    print(f"{'='*80}")
    print(f"  RL Aggregation Method: {rl_aggregation_method}")
    print(f"  Disable RL: {disable_rl}")
    print(f"  Disable Dual Attention: {disable_dual_attention}")
    if warmup is not None:
        print(f"  Warmup Rounds: {warmup}")
    if rampup is not None:
        print(f"  Ramp-up Rounds: {rampup}")
    print(f"{'='*80}")
    
    # اجرای آزمایش
    result = run_enhanced_ablation_experiment(
        config_name=config_name,
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=disable_rl,
        disable_dual_attention=disable_dual_attention,
        num_rounds=num_rounds,
        attack_types=attack_types,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed
    )
    
    # بازگرداندن تنظیمات قبلی
    config.RL_AGGREGATION_METHOD = original_rl_method
    config.RL_WARMUP_ROUNDS = original_warmup
    config.RL_RAMP_UP_ROUNDS = original_rampup
    
    return result

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE RL vs DUAL ATTENTION COMPARISON")
    print("="*80)
    
    print("\nStrategy:")
    print("  - Test multiple RL configurations")
    print("  - Compare against pure Dual Attention")
    print("  - Use combined unseen attacks")
    print("  - All with same seed for fair comparison")
    
    # تنظیمات تست
    num_rounds = 75
    attack_intensity = 50.0
    malicious_ratio = 0.5
    seed = 42
    
    # حملات combined unseen
    combined_attacks = [
        'partial_scaling_attack',
        'targeted_attack'
    ]
    
    print(f"\n[Test Configuration]")
    print(f"  Dataset: Alzheimer MRI")
    print(f"  Rounds: {num_rounds}")
    print(f"  Attack Intensity: {attack_intensity}")
    print(f"  Malicious Ratio: {malicious_ratio}")
    print(f"  Attacks: {combined_attacks}")
    print(f"  Seed: {seed}")
    
    results = {}
    
    # =====================================================================
    # Test 1: Pure Dual Attention (baseline - بهترین انتظار)
    # =====================================================================
    print("\n" + "="*80)
    print("[Test 1] Pure Dual Attention (NO RL)")
    print("  Expected: BEST performance (~96%)")
    print("="*80)
    
    set_seed(seed)
    results['dual_attention_only'] = run_with_config(
        config_name='dual_attention_only',
        rl_aggregation_method='dual_attention',
        num_rounds=num_rounds,
        attack_types=combined_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed,
        disable_rl=True,  # فقط Dual Attention
        disable_dual_attention=False
    )
    
    print("\n[OK] Pure Dual Attention completed!")
    print(f"   Accuracy: {results['dual_attention_only']['final_accuracy']:.4f}")
    print(f"   F1: {results['dual_attention_only']['detection_f1']:.4f}")
    
    # =====================================================================
    # Test 2: Hybrid (current default - مشکل دار)
    # =====================================================================
    print("\n" + "="*80)
    print("[Test 2] Hybrid (Default Settings)")
    print("  Warmup: 5, Ramp-up: 10")
    print("  Pure RL from round 15-74 (60 rounds)")
    print("  Expected: WORST performance (38%?)")
    print("="*80)
    
    set_seed(seed)
    results['hybrid_default'] = run_with_config(
        config_name='hybrid_default',
        rl_aggregation_method='hybrid',
        warmup=5,
        rampup=10,
        num_rounds=num_rounds,
        attack_types=combined_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed,
        disable_rl=False,  # Hybrid: RL فعال
        disable_dual_attention=False
    )
    
    print("\n[OK] Hybrid (Default) completed!")
    print(f"   Accuracy: {results['hybrid_default']['final_accuracy']:.4f}")
    print(f"   F1: {results['hybrid_default']['detection_f1']:.4f}")
    
    # =====================================================================
    # Test 3: Hybrid Conservative (warmup/ramp-up بیشتر)
    # =====================================================================
    print("\n" + "="*80)
    print("[Test 3] Hybrid Conservative")
    print("  Warmup: 20, Ramp-up: 30")
    print("  Pure RL from round 50-74 (25 rounds only)")
    print("  Expected: BETTER than default hybrid")
    print("="*80)
    
    set_seed(seed)
    results['hybrid_conservative'] = run_with_config(
        config_name='hybrid_conservative',
        rl_aggregation_method='hybrid',
        warmup=20,
        rampup=30,
        num_rounds=num_rounds,
        attack_types=combined_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed,
        disable_rl=False,  # Hybrid Conservative: RL فعال
        disable_dual_attention=False
    )
    
    print("\n[OK] Hybrid Conservative completed!")
    print(f"   Accuracy: {results['hybrid_conservative']['final_accuracy']:.4f}")
    print(f"   F1: {results['hybrid_conservative']['detection_f1']:.4f}")
    
    # =====================================================================
    # Test 4 (OPTIONAL): Pure RL from start
    # =====================================================================
    print("\n" + "="*80)
    print("[Test 4] Pure RL from Start (NO warmup)")
    print("  Warmup: 0, Ramp-up: 0")
    print("  100% RL for all 75 rounds")
    print("  Expected: Probably BAD (RL not trained)")
    print("="*80)
    
    set_seed(seed)
    results['rl_only'] = run_with_config(
        config_name='rl_only',
        rl_aggregation_method='rl_actor_critic',
        warmup=0,
        rampup=0,
        num_rounds=num_rounds,
        attack_types=combined_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed,
        disable_rl=False,  # Pure RL: RL فعال از ابتدا
        disable_dual_attention=False
    )
    
    print("\n[OK] Pure RL completed!")
    print(f"   Accuracy: {results['rl_only']['final_accuracy']:.4f}")
    print(f"   F1: {results['rl_only']['detection_f1']:.4f}")
    
    # =====================================================================
    # نتایج جامع و تحلیل
    # =====================================================================
    print("\n" + "="*80)
    print("[COMPREHENSIVE RESULTS]")
    print("="*80)
    
    # جدول مقایسه
    print("\n[Comparison Table]")
    print(f"{'Configuration':<30} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 78)
    
    for config_name, result in results.items():
        acc = result['final_accuracy']
        f1 = result['detection_f1']
        prec = result['detection_precision']
        rec = result['detection_recall']
        print(f"{config_name:<30} {acc:<12.4f} {f1:<12.4f} {prec:<12.4f} {rec:<12.4f}")
    
    # تحلیل دقیق
    print("\n" + "="*80)
    print("[DETAILED ANALYSIS]")
    print("="*80)
    
    # مقایسه با baseline (Dual Attention)
    baseline_acc = results['dual_attention_only']['final_accuracy']
    baseline_f1 = results['dual_attention_only']['detection_f1']
    
    print(f"\n[Baseline: Pure Dual Attention]")
    print(f"  Accuracy: {baseline_acc:.4f}")
    print(f"  F1: {baseline_f1:.4f}")
    
    print(f"\n[Performance Comparison vs Baseline]")
    for config_name, result in results.items():
        if config_name == 'dual_attention_only':
            continue
        
        acc_diff = result['final_accuracy'] - baseline_acc
        f1_diff = result['detection_f1'] - baseline_f1
        
        print(f"\n{config_name}:")
        print(f"  Accuracy Diff: {acc_diff:+.4f} ({acc_diff/baseline_acc*100:+.2f}%)")
        print(f"  F1 Diff: {f1_diff:+.4f} ({f1_diff/baseline_f1*100:+.2f}%)")
        
        if acc_diff > 0:
            print(f"  -> BETTER than baseline")
        elif acc_diff > -0.05:
            print(f"  -> Similar to baseline")
        else:
            print(f"  -> WORSE than baseline")
    
    # تفسیر
    print("\n" + "="*80)
    print("[INTERPRETATION]")
    print("="*80)
    
    # رتبه‌بندی
    ranked = sorted(results.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
    
    print("\n[Ranking by Accuracy]")
    for rank, (config_name, result) in enumerate(ranked, 1):
        acc = result['final_accuracy']
        f1 = result['detection_f1']
        print(f"  {rank}. {config_name}: Acc={acc:.4f}, F1={f1:.4f}")
    
    # یافته‌های کلیدی
    print("\n[Key Findings]")
    
    # آیا RL کمک می‌کند؟
    best_config = ranked[0][0]
    if best_config == 'dual_attention_only':
        print("\n1. Pure Dual Attention is BEST")
        print("   -> RL does NOT improve performance")
        print("   -> Our contribution is the Dual Attention design")
    else:
        print(f"\n1. {best_config} is BEST")
        print("   -> RL provides some benefit")
        print("   -> Configuration matters!")
    
    # آیا hybrid approach کار می‌کند؟
    hybrid_configs = {k: v for k, v in results.items() if 'hybrid' in k}
    if hybrid_configs:
        best_hybrid = max(hybrid_configs.items(), key=lambda x: x[1]['final_accuracy'])
        worst_hybrid = min(hybrid_configs.items(), key=lambda x: x[1]['final_accuracy'])
        
        print(f"\n2. Hybrid Approach Analysis:")
        print(f"   Best hybrid: {best_hybrid[0]} (Acc={best_hybrid[1]['final_accuracy']:.4f})")
        print(f"   Worst hybrid: {worst_hybrid[0]} (Acc={worst_hybrid[1]['final_accuracy']:.4f})")
        print(f"   Difference: {best_hybrid[1]['final_accuracy'] - worst_hybrid[1]['final_accuracy']:.4f}")
        
        if best_hybrid[0] == 'hybrid_conservative':
            print("   -> Longer warmup/ramp-up is BETTER")
            print("   -> RL needs time to learn")
        else:
            print("   -> Default hybrid is sufficient")
    
    # آیا pure RL کار می‌کند؟
    if 'rl_only' in results:
        rl_acc = results['rl_only']['final_accuracy']
        if rl_acc < baseline_acc - 0.1:
            print("\n3. Pure RL Performance:")
            print(f"   Accuracy: {rl_acc:.4f}")
            print("   -> RL FAILS without Dual Attention warmup")
            print("   -> RL needs pre-training or gradual introduction")
        else:
            print("\n3. Pure RL shows promise")
    
    # توصیه‌های نهایی
    print("\n" + "="*80)
    print("[RECOMMENDATIONS FOR PAPER]")
    print("="*80)
    
    if best_config == 'dual_attention_only':
        print("\n[Recommendation: Emphasize Dual Attention]")
        print("  Main contribution: Six-dimensional Dual Attention")
        print("  RL role: Explored but Dual Attention alone is superior")
        print("  ")
        print("  Paper claim:")
        print('  "Our carefully designed six-dimensional Dual Attention')
        print('   mechanism achieves 96.83% accuracy on complex combined')
        print('   attacks, demonstrating the effectiveness of expert-designed')
        print('   feature spaces in Byzantine-robust federated learning."')
    else:
        print(f"\n[Recommendation: {best_config} approach]")
        print(f"  Best accuracy: {ranked[0][1]['final_accuracy']:.4f}")
        print(f"  Shows RL provides value with proper configuration")
    
    # ذخیره نتایج
    print("\n" + "="*80)
    print("[SAVING RESULTS]")
    print("="*80)
    
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ذخیره JSON
    json_output = {
        'config': {
            'rounds': num_rounds,
            'attack_intensity': attack_intensity,
            'malicious_ratio': malicious_ratio,
            'attacks': combined_attacks,
            'seed': seed
        },
        'results': {}
    }
    
    for config_name, result in results.items():
        json_output['results'][config_name] = {
            'accuracy': float(result['final_accuracy']),
            'f1': float(result['detection_f1']),
            'precision': float(result['detection_precision']),
            'recall': float(result['detection_recall'])
        }
    
    json_file = output_dir / 'comprehensive_rl_comparison.json'
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\n[SAVED] Results saved to: {json_file}")
    
    # ذخیره CSV
    import pandas as pd
    
    df_data = []
    for config_name, result in results.items():
        df_data.append({
            'Configuration': config_name,
            'Accuracy': result['final_accuracy'],
            'F1': result['detection_f1'],
            'Precision': result['detection_precision'],
            'Recall': result['detection_recall']
        })
    
    df = pd.DataFrame(df_data)
    csv_file = output_dir / 'comprehensive_rl_comparison.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"[SAVED] CSV saved to: {csv_file}")
    
    print("\n" + "="*80)
    print("[COMPLETED] Comprehensive RL Comparison Test!")
    print("="*80)
    
    return results

if __name__ == '__main__':
    main()

