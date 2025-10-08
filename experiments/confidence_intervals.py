"""
Confidence Intervals Module
============================

Runs experiments with multiple random seeds to compute confidence intervals.
Addresses Reviewer 3's requirement for statistical variability assessment.

Runs 5 independent experiments with different seeds and reports:
- Mean ± Std for all metrics
- Confidence intervals (95%)
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds


def run_single_seed_experiment(seed, output_dir, num_rounds=25):
    """Run a single experiment with specific seed."""
    
    print(f"\n{'='*80}")
    print(f"🎲 SEED {seed}")
    print(f"{'='*80}\n")
    
    set_random_seeds(seed)
    
    # Load dataset
    root_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Create server
    server = Server()
    server.set_datasets(root_loader, test_dataset)
    server._pretrain_global_model()
    
    initial_accuracy = server.evaluate_model()
    
    # Create clients
    root_client_dataset, client_datasets = create_client_datasets(
        train_dataset=root_dataset,
        num_clients=NUM_CLIENTS,
        iid=not ENABLE_NON_IID,
        alpha=DIRICHLET_ALPHA if ENABLE_NON_IID else None
    )
    
    clients = []
    for i in range(NUM_CLIENTS):
        client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
        clients.append(client)
    
    server.add_clients(clients)
    
    # Configure malicious clients
    num_malicious = int(NUM_CLIENTS * 0.3)
    malicious_indices = np.random.choice(NUM_CLIENTS, num_malicious, replace=False)
    
    for i in malicious_indices:
        clients[i].is_malicious = True
        clients[i].set_attack_parameters(attack_type='scaling_attack', scaling_factor=10.0)
    
    # Train VAE
    root_gradients = server._collect_root_gradients()
    server.vae = server.train_vae(root_gradients, vae_epochs=VAE_EPOCHS)
    
    # Train
    training_errors, round_metrics = server.train(num_rounds=num_rounds)
    
    # Evaluate
    final_accuracy = server.evaluate_model()
    improvement = final_accuracy - initial_accuracy
    
    # Detection metrics
    total_tp, total_fp, total_fn = 0, 0, 0
    
    if round_metrics:
        for round_idx, round_data in round_metrics.items():
            if 'detection_results' in round_data:
                det_results = round_data['detection_results']
                total_tp += det_results.get('true_positives', 0)
                total_fp += det_results.get('false_positives', 0)
                total_fn += det_results.get('false_negatives', 0)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    result = {
        'seed': seed,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1_score,
        'malicious_indices': malicious_indices.tolist()
    }
    
    print(f"  Final Accuracy: {final_accuracy:.4f}")
    print(f"  Improvement: {improvement:.4f}")
    print(f"  Detection F1: {f1_score:.4f}")
    
    return result


def compute_confidence_intervals(data, confidence=0.95):
    """Compute mean, std, and confidence intervals."""
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample std
    n = len(data)
    
    # t-distribution for small samples
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_critical * (std / np.sqrt(n))
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_margin': margin_error,
        'n': n
    }


def run_multiple_seeds(output_dir, num_runs=5, seeds=None, num_rounds=25):
    """
    Run experiments with multiple seeds and compute confidence intervals.
    
    Args:
        output_dir: Output directory
        num_runs: Number of independent runs
        seeds: List of seeds (if None, use default)
        num_rounds: Number of training rounds per run
    
    Returns:
        Dictionary with aggregated results and confidence intervals
    """
    
    print(f"\n{'🔁'*40}")
    print(f"CONFIDENCE INTERVALS - MULTIPLE SEEDS")
    print(f"{'🔁'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default seeds
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024][:num_runs]
    
    print(f"Running {num_runs} independent experiments")
    print(f"Seeds: {seeds}\n")
    
    all_results = []
    
    # Run each seed
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*80}")
        print(f"RUN {i+1}/{num_runs}")
        print(f"{'#'*80}")
        
        result = run_single_seed_experiment(seed, output_dir, num_rounds=num_rounds)
        all_results.append(result)
    
    # Aggregate results
    print(f"\n{'='*80}")
    print(f"📊 AGGREGATED RESULTS WITH CONFIDENCE INTERVALS")
    print(f"{'='*80}\n")
    
    metrics = ['final_accuracy', 'improvement', 'detection_precision', 'detection_recall', 'detection_f1']
    
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        ci = compute_confidence_intervals(values)
        aggregated[metric] = ci
    
    # Print table
    print(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'95% CI':<25}")
    print(f"{'-'*80}")
    
    for metric, stats_dict in aggregated.items():
        metric_name = metric.replace('_', ' ').title()
        ci_str = f"[{stats_dict['ci_lower']:.4f}, {stats_dict['ci_upper']:.4f}]"
        print(f"{metric_name:<25} {stats_dict['mean']:<12.4f} {stats_dict['std']:<12.4f} {ci_str:<25}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'num_runs': num_runs,
        'seeds': seeds,
        'num_rounds': num_rounds,
        'individual_results': all_results,
        'aggregated_statistics': aggregated,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / f'confidence_intervals_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    import pandas as pd
    
    # Individual runs CSV
    runs_data = []
    for r in all_results:
        runs_data.append({
            'Seed': r['seed'],
            'Final_Accuracy': r['final_accuracy'],
            'Improvement': r['improvement'],
            'Detection_Precision': r['detection_precision'],
            'Detection_Recall': r['detection_recall'],
            'Detection_F1': r['detection_f1']
        })
    
    df_runs = pd.DataFrame(runs_data)
    runs_csv = output_dir / f'individual_runs_{timestamp}.csv'
    df_runs.to_csv(runs_csv, index=False)
    
    # Summary statistics CSV
    summary_data = []
    for metric, stats_dict in aggregated.items():
        summary_data.append({
            'Metric': metric,
            'Mean': stats_dict['mean'],
            'Std': stats_dict['std'],
            'CI_Lower_95': stats_dict['ci_lower'],
            'CI_Upper_95': stats_dict['ci_upper'],
            'CI_Margin': stats_dict['ci_margin'],
            'N_Runs': stats_dict['n']
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_csv = output_dir / f'confidence_intervals_summary_{timestamp}.csv'
    df_summary.to_csv(summary_csv, index=False)
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Confidence Intervals Across {num_runs} Runs', fontsize=16, fontweight='bold')
    
    metrics_plot = [
        ('final_accuracy', 'Final Accuracy', axes[0, 0]),
        ('improvement', 'Accuracy Improvement', axes[0, 1]),
        ('detection_precision', 'Detection Precision', axes[0, 2]),
        ('detection_recall', 'Detection Recall', axes[1, 0]),
        ('detection_f1', 'Detection F1-Score', axes[1, 1])
    ]
    
    for metric, title, ax in metrics_plot:
        values = [r[metric] for r in all_results]
        mean = aggregated[metric]['mean']
        ci_lower = aggregated[metric]['ci_lower']
        ci_upper = aggregated[metric]['ci_upper']
        
        # Individual runs
        ax.scatter(range(1, len(values)+1), values, alpha=0.6, s=100, label='Individual Runs')
        
        # Mean line
        ax.axhline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.4f}')
        
        # Confidence interval
        ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='green', label=f'95% CI')
        
        ax.set_xlabel('Run')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / f'confidence_intervals_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Confidence intervals computation completed!")
    print(f"📁 Results: {results_file}")
    print(f"📊 Individual runs CSV: {runs_csv}")
    print(f"📊 Summary CSV: {summary_csv}")
    print(f"📈 Plot: {plot_path}")
    
    return {
        'results_file': str(results_file),
        'runs_csv': str(runs_csv),
        'summary_csv': str(summary_csv),
        'plot_path': str(plot_path),
        'results': results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='experiments/results/confidence_intervals')
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--seeds', nargs='+', type=int, help='Custom seeds')
    
    args = parser.parse_args()
    
    run_multiple_seeds(
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        seeds=args.seeds,
        num_rounds=args.rounds
    )

