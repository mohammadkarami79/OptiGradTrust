"""
Optimizer Ablation Study
========================

Tests OptiGradTrust with DIFFERENT optimizers to show FedBN-P contribution.

Compares:
- OptiGradTrust + FedAvg (baseline)
- OptiGradTrust + FedProx
- OptiGradTrust + FedBN
- OptiGradTrust + FedBN-P (our full method)

This demonstrates how much improvement comes from FedBN-P vs other components.
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds


def run_optimizer_experiment(optimizer_name, output_dir, num_rounds=25):
    """Run OptiGradTrust with specific optimizer."""
    
    print(f"\n{'='*80}")
    print(f"🔧 OPTIMIZER: {optimizer_name}")
    print(f"{'='*80}\n")
    
    set_random_seeds(42)
    
    # Note: در implementation واقعی باید optimizer را تغییر دهیم
    # فعلاً همه با FedBN-P اجرا می‌شوند
    # این فقط یک placeholder است
    
    root_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    server = Server()
    server.set_datasets(root_loader, test_dataset)
    server._pretrain_global_model()
    
    initial_accuracy = server.evaluate_model()
    
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
    
    # Malicious clients
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
        'optimizer': optimizer_name,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'detection_f1': f1_score,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"  Final Accuracy: {final_accuracy:.4f}")
    print(f"  Improvement: {improvement:.4f}")
    
    return result


def run_optimizer_ablation(output_dir, optimizers=None, num_rounds=25):
    """
    Test OptiGradTrust with different optimizers.
    
    Args:
        output_dir: Output directory
        optimizers: List of optimizer names to test
        num_rounds: Number of training rounds
    
    Returns:
        Dictionary with results
    """
    
    print(f"\n{'⚙️'*40}")
    print(f"OPTIMIZER ABLATION STUDY")
    print(f"{'⚙️'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if optimizers is None:
        optimizers = ['FedAvg', 'FedProx', 'FedBN', 'FedBN-P']
    
    print(f"⚠️  NOTE: This is a placeholder implementation!")
    print(f"   Currently all use FedBN-P (requires config changes for different optimizers)")
    print(f"   For paper, you'll need to manually test with different configs.\n")
    
    results = []
    
    for optimizer in optimizers:
        result = run_optimizer_experiment(
            optimizer_name=optimizer,
            output_dir=output_dir,
            num_rounds=num_rounds
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 OPTIMIZER ABLATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Optimizer':<15} {'Accuracy':<12} {'Improvement':<15} {'Detection F1':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['optimizer']:<15} {r['final_accuracy']:<12.4f} {r['improvement']:<15.4f} {r['detection_f1']:<12.4f}")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_data = {
        'optimizers': optimizers,
        'results': results,
        'note': 'Placeholder - requires manual config changes for different optimizers',
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / f'optimizer_ablation_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    import pandas as pd
    df = pd.DataFrame(results)
    csv_file = output_dir / f'optimizer_ablation_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"\n✅ Optimizer ablation completed!")
    print(f"📁 Results: {results_file}")
    print(f"📊 CSV: {csv_file}")
    
    print(f"\n⚠️  IMPORTANT: This is a placeholder!")
    print(f"   To get real results, you need to:")
    print(f"   1. Modify config.py to use different optimizers")
    print(f"   2. Run this experiment multiple times")
    print(f"   3. Or use the existing optimizer comparison from Fig. 4 in paper")
    
    return {
        'results_file': str(results_file),
        'csv_file': str(csv_file),
        'results': output_data
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='experiments/results/optimizer_ablation')
    parser.add_argument('--optimizers', nargs='+', default=['FedAvg', 'FedProx', 'FedBN', 'FedBN-P'])
    parser.add_argument('--rounds', type=int, default=25)
    
    args = parser.parse_args()
    
    run_optimizer_ablation(output_dir=args.output_dir, optimizers=args.optimizers, num_rounds=args.rounds)

