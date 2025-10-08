"""
Scalability Tests Module
=========================

Tests system performance at different scales:
1. Varying number of clients: 10, 20, 50, 100
2. Varying adversarial ratios: 10%, 20%, 30%, 40%, 50%

Addresses Reviewers 2 & 3's concerns about scalability and performance limits.
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds


def run_scalability_experiment(num_clients, adversarial_ratio, output_dir, num_rounds=20):
    """Run experiment with specified scale parameters."""
    
    print(f"\n{'='*80}")
    print(f"⚡ SCALE: {num_clients} clients, {adversarial_ratio*100:.0f}% adversarial")
    print(f"{'='*80}\n")
    
    set_random_seeds(42)
    
    start_time = time.time()
    
    # Load dataset
    root_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Create server
    server = Server()
    server.set_datasets(root_loader, test_dataset)
    server._pretrain_global_model()
    
    initial_accuracy = server.evaluate_model()
    
    # Create client datasets
    root_client_dataset, client_datasets = create_client_datasets(
        train_dataset=root_dataset,
        num_clients=num_clients,
        iid=not ENABLE_NON_IID,
        alpha=DIRICHLET_ALPHA if ENABLE_NON_IID else None
    )
    
    # Create clients
    clients = []
    for i in range(num_clients):
        client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
        clients.append(client)
    
    server.add_clients(clients)
    
    # Configure malicious clients
    num_malicious = int(num_clients * adversarial_ratio)
    malicious_indices = np.random.choice(num_clients, num_malicious, replace=False)
    
    print(f"Malicious clients: {num_malicious}/{num_clients}")
    
    for i in malicious_indices:
        clients[i].is_malicious = True
        clients[i].set_attack_parameters(attack_type='scaling_attack', scaling_factor=10.0)
    
    # Train VAE
    root_gradients = server._collect_root_gradients()
    server.vae = server.train_vae(root_gradients, vae_epochs=min(VAE_EPOCHS, 15))  # Limit for scalability
    
    # Train
    train_start = time.time()
    training_errors, round_metrics = server.train(num_rounds=num_rounds)
    train_time = time.time() - train_start
    
    # Evaluate
    final_accuracy = server.evaluate_model()
    improvement = final_accuracy - initial_accuracy
    
    total_time = time.time() - start_time
    avg_round_time = train_time / num_rounds
    
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
        'num_clients': num_clients,
        'adversarial_ratio': adversarial_ratio,
        'num_malicious': num_malicious,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'detection_f1': f1_score,
        'total_time_seconds': total_time,
        'training_time_seconds': train_time,
        'avg_round_time_seconds': avg_round_time,
        'time_per_client_per_round': avg_round_time / num_clients,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n  Results:")
    print(f"    Final Accuracy: {final_accuracy:.4f}")
    print(f"    Detection F1: {f1_score:.4f}")
    print(f"    Total Time: {total_time:.2f}s")
    print(f"    Avg Round Time: {avg_round_time:.3f}s")
    print(f"    Time/Client/Round: {result['time_per_client_per_round']*1000:.2f}ms")
    
    return result


def test_varying_clients(output_dir, client_counts=None, num_rounds=20):
    """
    Test scalability with varying number of clients.
    
    Args:
        output_dir: Output directory
        client_counts: List of client counts to test
        num_rounds: Number of rounds per experiment
    
    Returns:
        Dictionary with results
    """
    
    print(f"\n{'📈'*40}")
    print(f"SCALABILITY TEST: VARYING NUMBER OF CLIENTS")
    print(f"{'📈'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if client_counts is None:
        client_counts = [10, 20, 50, 100]
    
    fixed_adversarial_ratio = 0.3  # 30% malicious
    
    results = []
    
    for i, num_clients in enumerate(client_counts):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i+1}/{len(client_counts)}: {num_clients} clients")
        print(f"{'='*80}")
        
        result = run_scalability_experiment(
            num_clients=num_clients,
            adversarial_ratio=fixed_adversarial_ratio,
            output_dir=output_dir,
            num_rounds=num_rounds
        )
        
        results.append(result)
    
    # Analysis
    print(f"\n{'='*80}")
    print(f"📊 SCALABILITY ANALYSIS - VARYING CLIENTS")
    print(f"{'='*80}\n")
    
    print(f"{'Clients':<10} {'Accuracy':<12} {'Det F1':<12} {'Round Time':<15} {'Per-Client Time':<18}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['num_clients']:<10} {r['final_accuracy']:<12.4f} {r['detection_f1']:<12.4f} "
              f"{r['avg_round_time_seconds']:<15.3f} {r['time_per_client_per_round']*1000:<18.2f}ms")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_data = {
        'experiment_type': 'varying_clients',
        'client_counts': client_counts,
        'fixed_adversarial_ratio': fixed_adversarial_ratio,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / f'scalability_clients_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # CSV
    import pandas as pd
    
    df = pd.DataFrame(results)
    csv_file = output_dir / f'scalability_clients_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scalability: Varying Number of Clients', fontsize=16, fontweight='bold')
    
    clients = [r['num_clients'] for r in results]
    accuracies = [r['final_accuracy'] for r in results]
    f1_scores = [r['detection_f1'] for r in results]
    round_times = [r['avg_round_time_seconds'] for r in results]
    per_client_times = [r['time_per_client_per_round']*1000 for r in results]
    
    # Accuracy vs clients
    axes[0, 0].plot(clients, accuracies, marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clients')
    axes[0, 0].set_ylabel('Final Accuracy')
    axes[0, 0].set_title('Accuracy vs Number of Clients')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Detection F1 vs clients
    axes[0, 1].plot(clients, f1_scores, marker='s', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Number of Clients')
    axes[0, 1].set_ylabel('Detection F1-Score')
    axes[0, 1].set_title('Detection Performance vs Number of Clients')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Round time vs clients
    axes[1, 0].plot(clients, round_times, marker='^', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Number of Clients')
    axes[1, 0].set_ylabel('Average Round Time (s)')
    axes[1, 0].set_title('Computational Time vs Number of Clients')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Per-client time vs clients
    axes[1, 1].plot(clients, per_client_times, marker='d', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Number of Clients')
    axes[1, 1].set_ylabel('Time per Client per Round (ms)')
    axes[1, 1].set_title('Per-Client Processing Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f'scalability_clients_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Varying clients test completed!")
    print(f"📁 Results: {results_file}")
    print(f"📊 CSV: {csv_file}")
    print(f"📈 Plot: {plot_path}")
    
    return {
        'results_file': str(results_file),
        'csv_file': str(csv_file),
        'plot_path': str(plot_path),
        'results': output_data
    }


def test_varying_adversarial_ratios(output_dir, ratios=None, num_rounds=20):
    """
    Test robustness with varying adversarial ratios.
    
    Args:
        output_dir: Output directory
        ratios: List of adversarial ratios to test
        num_rounds: Number of rounds per experiment
    
    Returns:
        Dictionary with results
    """
    
    print(f"\n{'⚠️'*40}")
    print(f"SCALABILITY TEST: VARYING ADVERSARIAL RATIOS")
    print(f"{'⚠️'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    fixed_num_clients = 10  # Fixed number of clients
    
    results = []
    
    for i, ratio in enumerate(ratios):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i+1}/{len(ratios)}: {ratio*100:.0f}% adversarial")
        print(f"{'='*80}")
        
        result = run_scalability_experiment(
            num_clients=fixed_num_clients,
            adversarial_ratio=ratio,
            output_dir=output_dir,
            num_rounds=num_rounds
        )
        
        results.append(result)
    
    # Analysis
    print(f"\n{'='*80}")
    print(f"📊 ROBUSTNESS ANALYSIS - VARYING ADVERSARIAL RATIOS")
    print(f"{'='*80}\n")
    
    print(f"{'Adv Ratio':<12} {'# Malicious':<15} {'Accuracy':<12} {'Det F1':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['adversarial_ratio']*100:.0f}%{'':<8} {r['num_malicious']:<15} "
              f"{r['final_accuracy']:<12.4f} {r['detection_f1']:<12.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_data = {
        'experiment_type': 'varying_adversarial_ratios',
        'ratios': ratios,
        'fixed_num_clients': fixed_num_clients,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / f'scalability_adversarial_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # CSV
    import pandas as pd
    
    df = pd.DataFrame(results)
    csv_file = output_dir / f'scalability_adversarial_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Robustness: Varying Adversarial Ratios', fontsize=16, fontweight='bold')
    
    ratios_pct = [r['adversarial_ratio']*100 for r in results]
    accuracies = [r['final_accuracy'] for r in results]
    f1_scores = [r['detection_f1'] for r in results]
    
    # Accuracy vs ratio
    axes[0].plot(ratios_pct, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    axes[0].set_xlabel('Adversarial Ratio (%)')
    axes[0].set_ylabel('Final Accuracy')
    axes[0].set_title('Accuracy vs Adversarial Ratio')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess')
    axes[0].legend()
    
    # Detection F1 vs ratio
    axes[1].plot(ratios_pct, f1_scores, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Adversarial Ratio (%)')
    axes[1].set_ylabel('Detection F1-Score')
    axes[1].set_title('Detection Performance vs Adversarial Ratio')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f'scalability_adversarial_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Varying adversarial ratios test completed!")
    print(f"📁 Results: {results_file}")
    print(f"📊 CSV: {csv_file}")
    print(f"📈 Plot: {plot_path}")
    
    return {
        'results_file': str(results_file),
        'csv_file': str(csv_file),
        'plot_path': str(plot_path),
        'results': output_data
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['clients', 'adversarial', 'both'], default='both')
    parser.add_argument('--output-dir', default='experiments/results/scalability')
    parser.add_argument('--rounds', type=int, default=20)
    
    args = parser.parse_args()
    
    if args.test in ['clients', 'both']:
        test_varying_clients(output_dir=args.output_dir, num_rounds=args.rounds)
    
    if args.test in ['adversarial', 'both']:
        test_varying_adversarial_ratios(output_dir=args.output_dir, num_rounds=args.rounds)

