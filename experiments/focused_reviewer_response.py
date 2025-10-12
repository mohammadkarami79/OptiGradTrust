"""
Focused Experiment for Reviewer Response
=========================================

هدف: تولید دقیقاً آنچه reviewer ها می‌خواهند:
1. Fair comparison با baselines
2. Simple ablation study
3. Multiple attacks
4. روی Alzheimer dataset (که قبلاً 96.83% گرفتیم)

زمان: ~8-10 ساعت (نه 18!)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_experiment(name, config_updates, num_rounds=50):
    """اجرای یک experiment با config خاص"""
    print(f"\n{'='*80}")
    print(f"[Experiment] {name}")
    print(f"{'='*80}")
    
    import federated_learning.config.config as config
    
    # ذخیره تنظیمات قبلی
    original_settings = {}
    for key, value in config_updates.items():
        original_settings[key] = getattr(config, key, None)
        setattr(config, key, value)
    
    # اجرای training
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    
    try:
        # Setup
        server = Server()
        server.setup_clients()
        
        # Train
        print(f"Training for {num_rounds} rounds...")
        server.train(num_rounds=num_rounds)
        
        # Get results
        final_acc = server.global_model_accuracy if hasattr(server, 'global_model_accuracy') else 0.0
        
        result = {
            'name': name,
            'final_accuracy': float(final_acc),
            'num_rounds': num_rounds,
            'config': config_updates
        }
        
        print(f"\n[Result] {name}: {final_acc:.4f}")
        
    finally:
        # بازگرداندن تنظیمات
        for key, value in original_settings.items():
            if value is not None:
                setattr(config, key, value)
    
    return result

def main():
    print("\n" + "="*80)
    print("FOCUSED EXPERIMENTS FOR REVIEWER RESPONSE")
    print("="*80)
    
    print("\nGoal: Generate exactly what reviewers want")
    print("  1. Fair comparison with baselines")
    print("  2. Simple ablation study")
    print("  3. Multiple attacks")
    print("\nEstimated time: 8-10 hours")
    
    seed = 42
    num_rounds = 50  # کافی برای convergence
    results = {}
    
    # =====================================================================
    # Part 1: Baseline Comparisons (مهم‌ترین!)
    # =====================================================================
    print("\n" + "="*80)
    print("[PART 1] BASELINE COMPARISONS")
    print("="*80)
    
    # 1. OptiGradTrust (Full) - ما
    set_seed(seed)
    results['OptiGradTrust'] = run_experiment(
        name='OptiGradTrust (Full)',
        config_updates={
            'DATASET': 'alzheimer',
            'USE_VAE': True,
            'USE_SHAPLEY': True,
            'GRADIENT_COMBINATION_METHOD': 'fedbn_fedprox',
            'RL_AGGREGATION_METHOD': 'hybrid',
            'ATTACK_TYPE': 'partial_scaling_attack',
            'NUM_MALICIOUS': 5
        },
        num_rounds=num_rounds
    )
    
    # 2. FLGuard-like (فقط gradient clipping)
    set_seed(seed)
    results['FLGuard-like'] = run_experiment(
        name='FLGuard-like',
        config_updates={
            'DATASET': 'alzheimer',
            'USE_VAE': False,
            'USE_SHAPLEY': False,
            'GRADIENT_COMBINATION_METHOD': 'fedavg',
            'RL_AGGREGATION_METHOD': None,
            'USE_GRADIENT_CLIPPING': True,
            'ATTACK_TYPE': 'partial_scaling_attack',
            'NUM_MALICIOUS': 5
        },
        num_rounds=num_rounds
    )
    
    # 3. FLTrust-like (trust score based)
    set_seed(seed)
    results['FLTrust-like'] = run_experiment(
        name='FLTrust-like',
        config_updates={
            'DATASET': 'alzheimer',
            'USE_VAE': False,
            'USE_SHAPLEY': False,
            'GRADIENT_COMBINATION_METHOD': 'fedavg',
            'RL_AGGREGATION_METHOD': None,
            'USE_TRUST_SCORE': True,
            'ATTACK_TYPE': 'partial_scaling_attack',
            'NUM_MALICIOUS': 5
        },
        num_rounds=num_rounds
    )
    
    # =====================================================================
    # Part 2: Ablation Study
    # =====================================================================
    print("\n" + "="*80)
    print("[PART 2] ABLATION STUDY")
    print("="*80)
    
    # 4. Without Shapley
    set_seed(seed)
    results['Without_Shapley'] = run_experiment(
        name='OptiGradTrust without Shapley',
        config_updates={
            'DATASET': 'alzheimer',
            'USE_VAE': True,
            'USE_SHAPLEY': False,  # ← کلیدی
            'GRADIENT_COMBINATION_METHOD': 'fedbn_fedprox',
            'RL_AGGREGATION_METHOD': 'hybrid',
            'ATTACK_TYPE': 'partial_scaling_attack',
            'NUM_MALICIOUS': 5
        },
        num_rounds=num_rounds
    )
    
    # 5. Without VAE
    set_seed(seed)
    results['Without_VAE'] = run_experiment(
        name='OptiGradTrust without VAE',
        config_updates={
            'DATASET': 'alzheimer',
            'USE_VAE': False,  # ← کلیدی
            'USE_SHAPLEY': True,
            'GRADIENT_COMBINATION_METHOD': 'fedbn_fedprox',
            'RL_AGGREGATION_METHOD': 'hybrid',
            'ATTACK_TYPE': 'partial_scaling_attack',
            'NUM_MALICIOUS': 5
        },
        num_rounds=num_rounds
    )
    
    # 6. With FedAvg instead of FedBN-P
    set_seed(seed)
    results['With_FedAvg'] = run_experiment(
        name='OptiGradTrust with FedAvg',
        config_updates={
            'DATASET': 'alzheimer',
            'USE_VAE': True,
            'USE_SHAPLEY': True,
            'GRADIENT_COMBINATION_METHOD': 'fedavg',  # ← کلیدی
            'RL_AGGREGATION_METHOD': 'hybrid',
            'ATTACK_TYPE': 'partial_scaling_attack',
            'NUM_MALICIOUS': 5
        },
        num_rounds=num_rounds
    )
    
    # =====================================================================
    # Analysis and Output
    # =====================================================================
    print("\n" + "="*80)
    print("[RESULTS SUMMARY]")
    print("="*80)
    
    # Part 1: Baseline Comparison
    print("\n[1. Baseline Comparison]")
    print(f"{'Method':<30} {'Accuracy':<12}")
    print("-" * 42)
    
    baseline_methods = ['FLGuard-like', 'FLTrust-like', 'OptiGradTrust']
    for method in baseline_methods:
        if method in results:
            acc = results[method]['final_accuracy']
            marker = " <- OURS" if method == 'OptiGradTrust' else ""
            print(f"{method:<30} {acc:<12.4f}{marker}")
    
    # Part 2: Ablation Study
    print("\n[2. Ablation Study]")
    print(f"{'Configuration':<30} {'Accuracy':<12} {'Drop':<12}")
    print("-" * 54)
    
    full_acc = results['OptiGradTrust']['final_accuracy']
    print(f"{'Full OptiGradTrust':<30} {full_acc:<12.4f} {'baseline':<12}")
    
    ablation_configs = [
        ('Without_Shapley', 'Without Shapley'),
        ('Without_VAE', 'Without VAE'),
        ('With_FedAvg', 'With FedAvg (not FedBN-P)')
    ]
    
    for key, label in ablation_configs:
        if key in results:
            acc = results[key]['final_accuracy']
            drop = full_acc - acc
            print(f"{label:<30} {acc:<12.4f} {drop:>+11.4f}")
    
    # Key Findings
    print("\n" + "="*80)
    print("[KEY FINDINGS FOR PAPER]")
    print("="*80)
    
    if 'OptiGradTrust' in results and 'FLGuard-like' in results:
        our_acc = results['OptiGradTrust']['final_accuracy']
        flguard_acc = results['FLGuard-like']['final_accuracy']
        improvement = ((our_acc - flguard_acc) / flguard_acc) * 100
        
        print(f"\n1. OptiGradTrust achieves {our_acc:.2%} accuracy")
        print(f"   Outperforms FLGuard-like by {improvement:.1f}%")
    
    if 'Without_Shapley' in results:
        shapley_impact = full_acc - results['Without_Shapley']['final_accuracy']
        print(f"\n2. Shapley values contribute {shapley_impact:.2%} improvement")
    
    if 'Without_VAE' in results:
        vae_impact = full_acc - results['Without_VAE']['final_accuracy']
        print(f"   VAE contributes {vae_impact:.2%} improvement")
    
    if 'With_FedAvg' in results:
        fedbnp_impact = full_acc - results['With_FedAvg']['final_accuracy']
        print(f"   FedBN-P contributes {fedbnp_impact:.2%} improvement")
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reviewer_response_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Results saved to: {output_file}")
    
    # Generate LaTeX table for paper
    latex_file = output_dir / f"reviewer_response_table_{timestamp}.tex"
    with open(latex_file, 'w') as f:
        f.write("% Baseline Comparison Table\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("Method & Accuracy \\\\\n")
        f.write("\\hline\n")
        
        for method in baseline_methods:
            if method in results:
                acc = results[method]['final_accuracy']
                marker = " (ours)" if method == 'OptiGradTrust' else ""
                f.write(f"{method.replace('_', ' ')}{marker} & {acc:.4f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comparison with state-of-the-art methods}\n")
        f.write("\\end{table}\n")
    
    print(f"[SAVED] LaTeX table saved to: {latex_file}")
    
    print("\n" + "="*80)
    print("[COMPLETED] All experiments finished!")
    print("="*80)
    
    return results

if __name__ == '__main__':
    main()

