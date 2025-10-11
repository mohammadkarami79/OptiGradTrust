"""
Quick Verification Test for Comprehensive RL Comparison
========================================================

این تست سریع (10 rounds) برای تأیید اینکه configurations مختلف
واقعاً نتایج متفاوت می‌دهند.

Expected runtime: ~30-40 minutes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main test function but modify settings
from experiments.test_comprehensive_rl_comparison import run_with_config, set_seed
import json
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("[QUICK VERIFICATION TEST]")
    print("="*80)
    
    print("\nGoal: Verify that different configurations produce different results")
    print("Rounds: 10 (instead of 75)")
    print("Expected time: ~30-40 minutes")
    
    # تنظیمات تست
    num_rounds = 10  # کم!
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
    print(f"  Rounds: {num_rounds} (QUICK)")
    print(f"  Attack Intensity: {attack_intensity}")
    print(f"  Malicious Ratio: {malicious_ratio}")
    print(f"  Attacks: {combined_attacks}")
    print(f"  Seed: {seed}")
    
    results = {}
    
    # Test 1: Pure Dual Attention
    print("\n" + "="*80)
    print("[Test 1] Pure Dual Attention (NO RL)")
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
        disable_rl=True,
        disable_dual_attention=False
    )
    
    print(f"\n[RESULT] Dual Attention: Acc={results['dual_attention_only']['final_accuracy']:.4f}")
    
    # Test 2: Hybrid Default
    print("\n" + "="*80)
    print("[Test 2] Hybrid (Default Settings)")
    print("  Warmup: 2, Ramp-up: 3 (adjusted for 10 rounds)")
    print("="*80)
    
    set_seed(seed)
    results['hybrid_default'] = run_with_config(
        config_name='hybrid_default',
        rl_aggregation_method='hybrid',
        warmup=2,  # Adjusted
        rampup=3,  # Adjusted
        num_rounds=num_rounds,
        attack_types=combined_attacks,
        malicious_ratio=malicious_ratio,
        attack_intensity=attack_intensity,
        seed=seed,
        disable_rl=False,
        disable_dual_attention=False
    )
    
    print(f"\n[RESULT] Hybrid Default: Acc={results['hybrid_default']['final_accuracy']:.4f}")
    
    # Test 3: Pure RL
    print("\n" + "="*80)
    print("[Test 3] Pure RL from Start")
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
        disable_rl=False,
        disable_dual_attention=False
    )
    
    print(f"\n[RESULT] Pure RL: Acc={results['rl_only']['final_accuracy']:.4f}")
    
    # Analysis
    print("\n" + "="*80)
    print("[VERIFICATION RESULTS]")
    print("="*80)
    
    acc_dual = results['dual_attention_only']['final_accuracy']
    acc_hybrid = results['hybrid_default']['final_accuracy']
    acc_rl = results['rl_only']['final_accuracy']
    
    print(f"\nAccuracies:")
    print(f"  Dual Attention: {acc_dual:.4f}")
    print(f"  Hybrid Default:  {acc_hybrid:.4f}")
    print(f"  Pure RL:         {acc_rl:.4f}")
    
    # Check if results are different
    all_same = (acc_dual == acc_hybrid == acc_rl)
    
    if all_same:
        print("\n" + "="*80)
        print("[FAILED] All results are IDENTICAL!")
        print("  Problem: Configurations are still running the same code path")
        print("  Action: Need to debug further")
        print("="*80)
        return False
    else:
        print("\n" + "="*80)
        print("[SUCCESS] Results are DIFFERENT!")
        print("  Dual vs Hybrid: {:.4f}".format(abs(acc_dual - acc_hybrid)))
        print("  Dual vs RL:     {:.4f}".format(abs(acc_dual - acc_rl)))
        print("  Hybrid vs RL:   {:.4f}".format(abs(acc_hybrid - acc_rl)))
        print("\n  The fix worked! Ready for full test.")
        print("="*80)
        
        # Save quick results
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        quick_results = {
            'test_type': 'quick_verification',
            'rounds': num_rounds,
            'results': {
                'dual_attention': float(acc_dual),
                'hybrid_default': float(acc_hybrid),
                'rl_only': float(acc_rl)
            },
            'verdict': 'SUCCESS - configurations produce different results'
        }
        
        json_file = output_dir / 'quick_verification_results.json'
        with open(json_file, 'w') as f:
            json.dump(quick_results, f, indent=2)
        
        print(f"\n[SAVED] Quick results: {json_file}")
        
        return True

if __name__ == '__main__':
    success = main()
    
    if success:
        print("\n" + "="*80)
        print("[NEXT STEP]")
        print("="*80)
        print("\nNow you can run the full 75-round test:")
        print("  python experiments\\test_comprehensive_rl_comparison.py")
        print("\nExpected time: ~20-24 hours")
        print("="*80)
    else:
        print("\n[DEBUG NEEDED] Please check the implementation.")

