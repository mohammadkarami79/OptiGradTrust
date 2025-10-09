"""
تست سریع Ablation Study v2
============================

این فایل یک تست بسیار سریع (5 rounds) اجرا می‌کند
تا مطمئن شویم همه چیز کار می‌کند.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.ablation_study_v2 import run_enhanced_ablation_experiment

def quick_test():
    """یک تست خیلی سریع با 5 rounds"""
    
    print("\n" + "="*80)
    print("🧪 QUICK TEST: Ablation Study v2")
    print("="*80)
    print("این تست فقط 2 scenario را با 5 rounds test می‌کند")
    print("="*80 + "\n")
    
    results = {}
    
    # Test 1: Baseline
    print("\n1️⃣  Testing BASELINE...")
    results['baseline'] = run_enhanced_ablation_experiment(
        config_name='test_baseline',
        num_rounds=5,
        attack_types=['scaling_attack'],
        malicious_ratio=0.3,
        attack_intensity=15.0,
        seed=42
    )
    
    # Test 2: Without RL
    print("\n2️⃣  Testing WITHOUT RL...")
    results['without_rl'] = run_enhanced_ablation_experiment(
        config_name='test_without_rl',
        disable_rl=True,
        num_rounds=5,
        attack_types=['scaling_attack'],
        malicious_ratio=0.3,
        attack_intensity=15.0,
        seed=42
    )
    
    # Compare
    print("\n" + "="*80)
    print("📊 QUICK TEST RESULTS")
    print("="*80)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['final_accuracy']:.4f}")
        print(f"  Detection F1: {result['detection_f1']:.4f}")
    
    # Check if there's a difference
    acc_diff = abs(results['baseline']['final_accuracy'] - results['without_rl']['final_accuracy'])
    f1_diff = abs(results['baseline']['detection_f1'] - results['without_rl']['detection_f1'])
    
    print("\n" + "="*80)
    if acc_diff > 0.001 or f1_diff > 0.01:
        print("✅ تفاوت قابل مشاهده است! پیاده‌سازی درست است.")
        print(f"   Accuracy difference: {acc_diff:.4f}")
        print(f"   F1 difference: {f1_diff:.4f}")
    else:
        print("⚠️  تفاوت خیلی کم است. ممکن است نیاز به:")
        print("   - Rounds بیشتر")
        print("   - حملات قوی‌تر")
        print("   - تنظیمات متفاوت")
    print("="*80)
    
    return results

if __name__ == "__main__":
    quick_test()

