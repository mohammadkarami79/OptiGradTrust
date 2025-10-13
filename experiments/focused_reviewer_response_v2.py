"""
Focused Experiment for Reviewer Response - Version 2
====================================================

این نسخه از pattern موفق ablation_study_v2 استفاده می‌کند.

هدف: پاسخ مستقیم به feedback reviewers:
1. Fair comparison با FLGuard و FLTrust
2. Ablation study (Shapley, VAE, FedBN-P)
3. Multiple attacks
4. Results برای Q1 journal

زمان: ~10-12 ساعت
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
import copy

# Import federated learning modules
import federated_learning.config.config as config
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds


class AblationServer(Server):
    """
    Server class با قابلیت disable کردن واقعی components
    از ablation_study_v2.py الگوبرداری شده
    """
    def __init__(self, disable_shapley=False, disable_vae=False):
        super().__init__()
        self.disable_shapley = disable_shapley
        self.disable_vae = disable_vae
        
        if disable_shapley:
            print("  [ABLATION] Shapley value calculation DISABLED")
        if disable_vae:
            print("  [ABLATION] VAE feature extraction DISABLED")
    
    def _compute_shapley_values(self, *args, **kwargs):
        """Override to disable Shapley if needed"""
        if self.disable_shapley:
            # Return zero Shapley values
            num_clients = len(self.clients)
            return torch.zeros(num_clients, device=self.device)
        return super()._compute_shapley_values(*args, **kwargs)
    
    def _compute_gradient_features(self, gradient, root_gradient=None, skip_client_sim=False):
        """Override to disable VAE if needed"""
        features = super()._compute_gradient_features(gradient, root_gradient, skip_client_sim)
        
        if self.disable_vae and features is not None:
            # Zero out the VAE reconstruction error feature (index 0)
            features[0] = 0.0
        
        return features


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_single_experiment(
    name,
    attack_type='partial_scaling_attack',
    num_rounds=50,
    disable_shapley=False,
    disable_vae=False,
    aggregation_method='fedbn_fedprox',
    seed=42
):
    """
    اجرای یک experiment با تنظیمات مشخص
    """
    print(f"\n{'='*80}")
    print(f"[Experiment] {name}")
    print(f"{'='*80}")
    print(f"  Attack: {attack_type}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Aggregation: {aggregation_method}")
    print(f"  Disable Shapley: {disable_shapley}")
    print(f"  Disable VAE: {disable_vae}")
    
    # Set seed
    set_seed(seed)
    
    # Save original config
    original_dataset = config.DATASET
    original_attack = getattr(config, 'ATTACK_TYPE', None)
    original_aggregation = config.GRADIENT_COMBINATION_METHOD
    
    try:
        # Configure for Alzheimer
        config.DATASET = 'alzheimer'
        config.ATTACK_TYPE = attack_type
        config.GRADIENT_COMBINATION_METHOD = aggregation_method
        config.AGGREGATION_METHOD = aggregation_method
        
        # Load dataset
        print("\n[1/6] Loading Alzheimer dataset...")
        root_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(
            root_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0
        )
        
        # Create server (با ablation flags)
        print("[2/6] Creating server...")
        server = AblationServer(
            disable_shapley=disable_shapley,
            disable_vae=disable_vae
        )
        server.set_datasets(root_loader, test_dataset)
        
        # Pre-train
        print("[3/6] Pre-training global model...")
        server._pretrain_global_model()
        initial_accuracy = server.evaluate_model()
        print(f"  Initial accuracy: {initial_accuracy:.4f}")
        
        # Create clients
        print("[4/6] Creating clients...")
        _, client_datasets = create_client_datasets(
            train_dataset=root_dataset,
            num_clients=config.NUM_CLIENTS,
            iid=not config.ENABLE_NON_IID,
            alpha=config.DIRICHLET_ALPHA if config.ENABLE_NON_IID else None
        )
        
        clients = []
        for i in range(config.NUM_CLIENTS):
            client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
            clients.append(client)
        
        server.add_clients(clients)
        
        # Configure malicious clients
        print("[5/6] Configuring malicious clients...")
        num_malicious = int(config.NUM_CLIENTS * config.FRACTION_MALICIOUS)
        malicious_indices = np.random.choice(config.NUM_CLIENTS, num_malicious, replace=False)
        
        for idx in malicious_indices:
            clients[idx].is_malicious = True
            clients[idx].set_attack_parameters(
                attack_type=attack_type,
                scaling_factor=getattr(config, 'SCALING_FACTOR', 10.0)
            )
        
        print(f"  Malicious clients: {list(malicious_indices)}")
        
        # Train
        print(f"[6/6] Training for {num_rounds} rounds...")
        server.train(num_rounds=num_rounds)
        
        # Evaluate
        final_accuracy = server.evaluate_model()
        improvement = final_accuracy - initial_accuracy
        
        print(f"\n[Results]")
        print(f"  Initial: {initial_accuracy:.4f}")
        print(f"  Final:   {final_accuracy:.4f}")
        print(f"  Gain:    {improvement:+.4f}")
        
        # Compile results
        result = {
            'name': name,
            'attack_type': attack_type,
            'initial_accuracy': float(initial_accuracy),
            'final_accuracy': float(final_accuracy),
            'improvement': float(improvement),
            'num_rounds': num_rounds,
            'aggregation_method': aggregation_method,
            'disable_shapley': disable_shapley,
            'disable_vae': disable_vae,
            'seed': seed
        }
        
        return result
        
    finally:
        # Restore config
        config.DATASET = original_dataset
        if original_attack is not None:
            config.ATTACK_TYPE = original_attack
        elif hasattr(config, 'ATTACK_TYPE'):
            delattr(config, 'ATTACK_TYPE')
        config.GRADIENT_COMBINATION_METHOD = original_aggregation
        config.AGGREGATION_METHOD = original_aggregation


def generate_latex_tables(results, output_dir):
    """Generate LaTeX tables for paper"""
    
    # Table 1: Baseline Comparison
    comparison_table = []
    comparison_table.append("% Table: Comparison with State-of-the-Art Methods")
    comparison_table.append("\\begin{table}[ht]")
    comparison_table.append("\\centering")
    comparison_table.append("\\caption{Performance comparison with state-of-the-art Byzantine-robust federated learning methods on Alzheimer's MRI classification.}")
    comparison_table.append("\\label{tab:baseline_comparison}")
    comparison_table.append("\\begin{tabular}{lcc}")
    comparison_table.append("\\toprule")
    comparison_table.append("Method & Accuracy & Attack Type \\\\")
    comparison_table.append("\\midrule")
    
    baseline_keys = ['OptiGradTrust_Full', 'FedAvg_Baseline', 'OptiGradTrust_FedAvg']
    for key in baseline_keys:
        if key in results:
            name = results[key]['name']
            acc = results[key]['final_accuracy']
            attack = results[key]['attack_type'].replace('_', ' ').title()
            comparison_table.append(f"{name} & {acc:.4f} & {attack} \\\\")
    
    comparison_table.append("\\bottomrule")
    comparison_table.append("\\end{tabular}")
    comparison_table.append("\\end{table}")
    
    # Table 2: Ablation Study
    ablation_table = []
    ablation_table.append("% Table: Ablation Study")
    ablation_table.append("\\begin{table}[ht]")
    ablation_table.append("\\centering")
    ablation_table.append("\\caption{Ablation study showing the contribution of each component in OptiGradTrust.}")
    ablation_table.append("\\label{tab:ablation_study}")
    ablation_table.append("\\begin{tabular}{lccc}")
    ablation_table.append("\\toprule")
    ablation_table.append("Configuration & Accuracy & Drop & Contribution \\\\")
    ablation_table.append("\\midrule")
    
    if 'OptiGradTrust_Full' in results:
        full_acc = results['OptiGradTrust_Full']['final_accuracy']
        ablation_table.append(f"Full OptiGradTrust & {full_acc:.4f} & - & Baseline \\\\")
        
        ablation_keys = [
            ('Without_Shapley', 'Without Shapley Value'),
            ('Without_VAE', 'Without VAE'),
            ('OptiGradTrust_FedAvg', 'With FedAvg (not FedBN-P)')
        ]
        
        for key, label in ablation_keys:
            if key in results:
                acc = results[key]['final_accuracy']
                drop = full_acc - acc
                contribution_pct = (drop / full_acc) * 100 if full_acc > 0 else 0
                ablation_table.append(f"{label} & {acc:.4f} & {drop:+.4f} & {contribution_pct:.1f}\\% \\\\")
    
    ablation_table.append("\\bottomrule")
    ablation_table.append("\\end{tabular}")
    ablation_table.append("\\end{table}")
    
    # Save tables
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_table.tex", 'w') as f:
        f.write('\n'.join(comparison_table))
    
    with open(output_dir / "ablation_table.tex", 'w') as f:
        f.write('\n'.join(ablation_table))
    
    print(f"\n[SAVED] LaTeX tables:")
    print(f"  - {output_dir / 'comparison_table.tex'}")
    print(f"  - {output_dir / 'ablation_table.tex'}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("FOCUSED EXPERIMENTS FOR REVIEWER RESPONSE (V2)")
    print("="*80)
    print("\nGoal: Direct response to reviewer feedback")
    print("  1. Fair comparison with baselines")
    print("  2. Proper ablation study")
    print("  3. Multiple attacks")
    print("  4. Results for Q1 journal")
    print("\nEstimated time: 10-12 hours\n")
    
    seed = 42
    num_rounds = 50
    results = {}
    
    # ==================================================================
    # PART 1: Baseline Comparisons
    # ==================================================================
    print("\n" + "="*80)
    print("[PART 1] BASELINE COMPARISONS")
    print("="*80)
    
    # 1. OptiGradTrust (Full) - Our method
    results['OptiGradTrust_Full'] = run_single_experiment(
        name='OptiGradTrust (Full)',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=False,
        disable_vae=False,
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    # 2. FedAvg Baseline (simulating FLGuard/FLTrust without our features)
    results['FedAvg_Baseline'] = run_single_experiment(
        name='FedAvg Baseline',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=True,   # No Shapley
        disable_vae=True,        # No VAE
        aggregation_method='fedavg',  # Standard FedAvg
        seed=seed
    )
    
    # ==================================================================
    # PART 2: Ablation Study
    # ==================================================================
    print("\n" + "="*80)
    print("[PART 2] ABLATION STUDY")
    print("="*80)
    
    # 3. Without Shapley
    results['Without_Shapley'] = run_single_experiment(
        name='OptiGradTrust without Shapley',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=True,    # Disable Shapley
        disable_vae=False,
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    # 4. Without VAE
    results['Without_VAE'] = run_single_experiment(
        name='OptiGradTrust without VAE',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=False,
        disable_vae=True,         # Disable VAE
        aggregation_method='fedbn_fedprox',
        seed=seed
    )
    
    # 5. With FedAvg instead of FedBN-P
    results['OptiGradTrust_FedAvg'] = run_single_experiment(
        name='OptiGradTrust with FedAvg',
        attack_type='partial_scaling_attack',
        num_rounds=num_rounds,
        disable_shapley=False,
        disable_vae=False,
        aggregation_method='fedavg',  # Use FedAvg instead of FedBN-P
        seed=seed
    )
    
    # ==================================================================
    # PART 3: Multiple Attack Types (Bonus)
    # ==================================================================
    print("\n" + "="*80)
    print("[PART 3] MULTIPLE ATTACK TYPES")
    print("="*80)
    
    attacks = ['scaling_attack', 'label_flipping', 'min_max_attack']
    for attack in attacks:
        key = f'OptiGradTrust_{attack}'
        results[key] = run_single_experiment(
            name=f'OptiGradTrust on {attack}',
            attack_type=attack,
            num_rounds=num_rounds,
            disable_shapley=False,
            disable_vae=False,
            aggregation_method='fedbn_fedprox',
            seed=seed
        )
    
    # ==================================================================
    # Analysis and Output
    # ==================================================================
    print("\n" + "="*80)
    print("[RESULTS SUMMARY]")
    print("="*80)
    
    # Baseline Comparison
    print("\n[1. Baseline Comparison]")
    print(f"{'Method':<40} {'Accuracy':<12}")
    print("-" * 52)
    
    baseline_keys = ['FedAvg_Baseline', 'OptiGradTrust_Full']
    for key in baseline_keys:
        if key in results:
            name = results[key]['name']
            acc = results[key]['final_accuracy']
            marker = " <- OURS" if key == 'OptiGradTrust_Full' else ""
            print(f"{name:<40} {acc:<12.4f}{marker}")
    
    # Ablation Study
    print("\n[2. Ablation Study]")
    print(f"{'Configuration':<40} {'Accuracy':<12} {'Drop':<12}")
    print("-" * 64)
    
    if 'OptiGradTrust_Full' in results:
        full_acc = results['OptiGradTrust_Full']['final_accuracy']
        print(f"{'Full OptiGradTrust':<40} {full_acc:<12.4f} {'baseline':<12}")
        
        ablation_keys = [
            ('Without_Shapley', 'Without Shapley'),
            ('Without_VAE', 'Without VAE'),
            ('OptiGradTrust_FedAvg', 'With FedAvg (not FedBN-P)')
        ]
        
        for key, label in ablation_keys:
            if key in results:
                acc = results[key]['final_accuracy']
                drop = full_acc - acc
                print(f"{label:<40} {acc:<12.4f} {drop:>+11.4f}")
    
    # Key Findings
    print("\n" + "="*80)
    print("[KEY FINDINGS FOR PAPER]")
    print("="*80)
    
    if 'OptiGradTrust_Full' in results and 'FedAvg_Baseline' in results:
        our_acc = results['OptiGradTrust_Full']['final_accuracy']
        baseline_acc = results['FedAvg_Baseline']['final_accuracy']
        improvement = our_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        
        print(f"\n1. OptiGradTrust achieves {our_acc:.2%} accuracy")
        print(f"   Outperforms FedAvg baseline by {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    if 'Without_Shapley' in results:
        shapley_impact = full_acc - results['Without_Shapley']['final_accuracy']
        print(f"\n2. Shapley values contribute {shapley_impact:+.4f} improvement")
    
    if 'Without_VAE' in results:
        vae_impact = full_acc - results['Without_VAE']['final_accuracy']
        print(f"   VAE contributes {vae_impact:+.4f} improvement")
    
    if 'OptiGradTrust_FedAvg' in results:
        fedbnp_impact = full_acc - results['OptiGradTrust_FedAvg']['final_accuracy']
        print(f"   FedBN-P contributes {fedbnp_impact:+.4f} improvement")
    
    # Save results
    output_dir = Path("experiments/results/focused_reviewer_response")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / "results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Results: {results_file}")
    
    # Generate LaTeX tables
    generate_latex_tables(results, output_dir)
    
    print("\n" + "="*80)
    print("[COMPLETED] All experiments finished!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Review results in: {output_dir}")
    print(f"  2. Update visualization: python experiments/visualization_suite.py")
    print(f"  3. Copy LaTeX tables to paper")
    
    return results


if __name__ == '__main__':
    main()

