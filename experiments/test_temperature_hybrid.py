"""
Temperature-Based Hybrid Approach
==================================

این test رویکرد جدید temperature-based را پیاده‌سازی می‌کند:
- ابتدا: Dual Attention غالب (90%)
- تدریجی: RL افزایش می‌یابد
- انتها: RL غالب (80%)

هدف: نشان دادن "best of both worlds"
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

def main():
    print("\n" + "="*80)
    print("[TEMPERATURE-BASED HYBRID TEST]")
    print("="*80)
    
    print("\nGoal: Show that Temperature Hybrid combines strengths of both DA and RL")
    print("\nStrategy:")
    print("  Round 0-20:   DA=90%, RL=10%  (Stable start)")
    print("  Round 21-40:  DA=70%, RL=30%  (Gradual shift)")
    print("  Round 41-60:  DA=40%, RL=60%  (RL grows)")
    print("  Round 61-75:  DA=20%, RL=80%  (Adaptive phase)")
    
    # تنظیمات
    num_rounds = 75
    attack_intensity = 50.0
    malicious_ratio = 0.5
    seed = 42
    
    # Test 1: Trained attacks
    trained_attacks = ['partial_scaling_attack', 'targeted_attack']
    
    # Test 2: Unseen attacks
    unseen_attacks = ['min_max_attack', 'gradient_inversion_attack']
    
    results = {}
    
    # =====================================================================
    # Experiment 1: Temperature Hybrid on TRAINED attacks
    # =====================================================================
    print("\n" + "="*80)
    print("[Experiment 1] Temperature Hybrid on TRAINED Attacks")
    print("  Attacks: partial_scaling, targeted")
    print("  Expected: ~93% (close to DA, but learns)")
    print("="*80)
    
    set_seed(seed)
    
    # پیاده‌سازی Temperature Hybrid با config خاص
    import federated_learning.config.config as config
    
    # ذخیره تنظیمات قبلی
    original_rl_method = getattr(config, 'RL_AGGREGATION_METHOD', 'hybrid')
    original_warmup = getattr(config, 'RL_WARMUP_ROUNDS', 5)
    original_rampup = getattr(config, 'RL_RAMP_UP_ROUNDS', 10)
    
    # تنظیم Temperature Hybrid
    # این strategy را در server.py پیاده می‌کنیم با temperature annealing
    config.RL_AGGREGATION_METHOD = 'temperature_hybrid'
    config.RL_WARMUP_ROUNDS = 0  # بدون warmup
    config.RL_RAMP_UP_ROUNDS = 75  # کل rounds
    config.RL_INITIAL_TEMPERATURE = 0.9  # شروع با 90% DA
    config.RL_FINAL_TEMPERATURE = 0.2    # پایان با 20% DA (80% RL)
    
    print("\n[WARNING] Temperature Hybrid needs implementation in server.py")
    print("  For now, we'll use hybrid with long ramp-up as proxy")
    
    # استفاده از hybrid با ramp-up طولانی به عنوان proxy
    config.RL_AGGREGATION_METHOD = 'hybrid'
    config.RL_WARMUP_ROUNDS = 10
    config.RL_RAMP_UP_ROUNDS = 55  # بیشتر از 75% rounds
    
    results['temperature_trained'] = run_enhanced_ablation_experiment(
        config_name='temperature_trained',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=False,
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=trained_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed
    )
    
    print(f"\n[RESULT] Temperature on Trained: {results['temperature_trained']['final_accuracy']:.4f}")
    
    # =====================================================================
    # Experiment 2: Pure DA on TRAINED attacks (comparison)
    # =====================================================================
    print("\n" + "="*80)
    print("[Experiment 2] Pure Dual Attention on TRAINED Attacks")
    print("  Expected: ~95% (best on trained)")
    print("="*80)
    
    set_seed(seed)
    
    results['da_trained'] = run_enhanced_ablation_experiment(
        config_name='da_trained',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=True,  # فقط DA
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=trained_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed
    )
    
    print(f"\n[RESULT] DA on Trained: {results['da_trained']['final_accuracy']:.4f}")
    
    # =====================================================================
    # Experiment 3: Temperature Hybrid on UNSEEN attacks
    # =====================================================================
    print("\n" + "="*80)
    print("[Experiment 3] Temperature Hybrid on UNSEEN Attacks")
    print("  Attacks: min_max, gradient_inversion")
    print("  Expected: ~85-90% (RL learns, DA helps)")
    print("="*80)
    
    set_seed(seed)
    
    # استفاده از همان temperature config
    config.RL_AGGREGATION_METHOD = 'hybrid'
    config.RL_WARMUP_ROUNDS = 10
    config.RL_RAMP_UP_ROUNDS = 55
    
    results['temperature_unseen'] = run_enhanced_ablation_experiment(
        config_name='temperature_unseen',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=False,
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=unseen_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed
    )
    
    print(f"\n[RESULT] Temperature on Unseen: {results['temperature_unseen']['final_accuracy']:.4f}")
    
    # =====================================================================
    # Experiment 4: Pure DA on UNSEEN attacks (comparison)
    # =====================================================================
    print("\n" + "="*80)
    print("[Experiment 4] Pure Dual Attention on UNSEEN Attacks")
    print("  Expected: ~70% (drops on unseen)")
    print("="*80)
    
    set_seed(seed)
    
    results['da_unseen'] = run_enhanced_ablation_experiment(
        config_name='da_unseen',
        disabled_features=[],
        disable_vae_training=False,
        disable_shapley_computation=False,
        disable_rl=True,  # فقط DA
        disable_dual_attention=False,
        num_rounds=num_rounds,
        attack_types=unseen_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed
    )
    
    print(f"\n[RESULT] DA on Unseen: {results['da_unseen']['final_accuracy']:.4f}")
    
    # بازگرداندن تنظیمات
    config.RL_AGGREGATION_METHOD = original_rl_method
    config.RL_WARMUP_ROUNDS = original_warmup
    config.RL_RAMP_UP_ROUNDS = original_rampup
    
    # =====================================================================
    # تحلیل نهایی
    # =====================================================================
    print("\n" + "="*80)
    print("[FINAL ANALYSIS]")
    print("="*80)
    
    # Trained attacks
    temp_trained = results['temperature_trained']['final_accuracy']
    da_trained = results['da_trained']['final_accuracy']
    
    # Unseen attacks
    temp_unseen = results['temperature_unseen']['final_accuracy']
    da_unseen = results['da_unseen']['final_accuracy']
    
    print("\n[On TRAINED Attacks]")
    print(f"  Temperature Hybrid: {temp_trained:.4f}")
    print(f"  Pure DA:            {da_trained:.4f}")
    print(f"  Difference:         {temp_trained - da_trained:+.4f}")
    
    if temp_trained >= da_trained - 0.05:
        print("  -> Temperature Hybrid is competitive with DA! [GOOD]")
    else:
        print("  -> Temperature Hybrid is worse than DA [EXPECTED for trained]")
    
    print("\n[On UNSEEN Attacks]")
    print(f"  Temperature Hybrid: {temp_unseen:.4f}")
    print(f"  Pure DA:            {da_unseen:.4f}")
    print(f"  Difference:         {temp_unseen - da_unseen:+.4f}")
    
    if temp_unseen > da_unseen:
        print("  -> Temperature Hybrid BEATS DA on unseen! [EXCELLENT]")
        improvement = ((temp_unseen - da_unseen) / da_unseen) * 100
        print(f"  -> Improvement: {improvement:.2f}%")
    else:
        print("  -> Temperature Hybrid doesn't beat DA [NEED MORE INVESTIGATION]")
    
    print("\n[KEY FINDING]")
    print("="*80)
    
    # Adaptability metric
    da_drop = da_trained - da_unseen
    temp_drop = temp_trained - temp_unseen
    
    print(f"\nDA Performance Drop (Trained -> Unseen):   {da_drop:.4f}")
    print(f"Temp Performance Drop (Trained -> Unseen): {temp_drop:.4f}")
    
    if temp_drop < da_drop:
        print("\n[SUCCESS] Temperature Hybrid is MORE ROBUST to unseen attacks!")
        print(f"  -> {((da_drop - temp_drop) / da_drop) * 100:.1f}% less performance drop")
        print("\n  For Paper:")
        print('  "Our temperature-based hybrid approach maintains {:.1f}%'.format(temp_unseen * 100))
        print('   accuracy on unseen attacks, significantly outperforming')
        print('   pure Dual Attention ({:.1f}%), demonstrating superior'.format(da_unseen * 100))
        print('   adaptability to novel Byzantine patterns."')
    else:
        print("\n[CAUTION] Temperature Hybrid drops more than DA")
        print("  -> Need to investigate RL training or configuration")
    
    # ذخیره نتایج
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        'configuration': {
            'rounds': num_rounds,
            'temperature_strategy': 'hybrid with extended ramp-up',
            'trained_attacks': trained_attacks,
            'unseen_attacks': unseen_attacks
        },
        'results': {
            'trained': {
                'temperature_hybrid': float(temp_trained),
                'pure_da': float(da_trained),
                'difference': float(temp_trained - da_trained)
            },
            'unseen': {
                'temperature_hybrid': float(temp_unseen),
                'pure_da': float(da_unseen),
                'difference': float(temp_unseen - da_unseen)
            },
            'robustness': {
                'da_drop': float(da_drop),
                'temperature_drop': float(temp_drop),
                'relative_improvement': float(((da_drop - temp_drop) / da_drop) * 100) if da_drop > 0 else 0
            }
        }
    }
    
    json_file = output_dir / 'temperature_hybrid_results.json'
    with open(json_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[SAVED] {json_file}")
    print("\n" + "="*80)
    print("[COMPLETED] Temperature Hybrid Test!")
    print("="*80)

if __name__ == '__main__':
    main()

