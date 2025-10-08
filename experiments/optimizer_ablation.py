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


class OptimizerServer(Server):
    """Server with configurable optimizer/aggregation method."""
    
    def __init__(self, aggregation_method='fedbn_fedprox'):
        super().__init__()
        self.aggregation_method = aggregation_method
    
    def _aggregate_gradients(self, client_gradients, client_weights=None):
        """Override to use specific aggregation method."""
        from federated_learning.training.aggregators import (
            FedAvgAggregator, 
            FedProxAggregator, 
            FedBnAggregator,
            FedBnFedProxAggregator
        )
        
        # Select aggregator based on method
        aggregator_map = {
            'fedavg': FedAvgAggregator(),
            'fedprox': FedProxAggregator(),
            'fedbn': FedBnAggregator(),
            'fedbn_fedprox': FedBnFedProxAggregator()
        }
        
        aggregator = aggregator_map.get(self.aggregation_method.lower(), FedBnFedProxAggregator())
        
        # Use the aggregator
        aggregated_gradient = aggregator.aggregate_gradients(client_gradients, client_weights)
        
        return aggregated_gradient


def run_optimizer_experiment(optimizer_name, output_dir, num_rounds=25):
    """Run OptiGradTrust with specific optimizer."""
    
    print(f"\n{'='*80}")
    print(f"🔧 OPTIMIZER: {optimizer_name}")
    print(f"{'='*80}\n")
    
    set_random_seeds(42)
    
    # Map optimizer names to aggregation methods
    optimizer_map = {
        'FedAvg': 'fedavg',
        'FedProx': 'fedprox',
        'FedBN': 'fedbn',
        'FedBN-P': 'fedbn_fedprox'
    }
    
    aggregation_method = optimizer_map.get(optimizer_name, 'fedbn_fedprox')
    
    print(f"📌 Using aggregation method: {aggregation_method}")
    print(f"📌 Trust mechanism: ACTIVE (OptiGradTrust framework)")
    print(f"📌 This shows the contribution of {optimizer_name} within OptiGradTrust\n")
    
    root_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Use custom server with specific aggregation method
    server = OptimizerServer(aggregation_method=aggregation_method)
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
    
    print(f"🎯 Malicious clients: {list(malicious_indices)}")
    
    for i in malicious_indices:
        clients[i].is_malicious = True
        clients[i].set_attack_parameters(attack_type='scaling_attack', scaling_factor=10.0)
    
    # Train VAE
    print(f"\n🔧 Training VAE...")
    root_gradients = server._collect_root_gradients()
    server.vae = server.train_vae(root_gradients, vae_epochs=VAE_EPOCHS)
    
    # Train
    print(f"\n🚀 Starting federated training with {optimizer_name}...")
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
        'aggregation_method': aggregation_method,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1_score,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'malicious_indices': malicious_indices.tolist(),
        'num_rounds': num_rounds,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n✅ Results:")
    print(f"   Final Accuracy: {final_accuracy:.4f}")
    print(f"   Improvement: {improvement:.4f}")
    print(f"   Detection F1: {f1_score:.4f}")
    
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
    
    print(f"📝 Testing {len(optimizers)} optimizers: {', '.join(optimizers)}")
    print(f"🎯 Each test uses OptiGradTrust framework (trust mechanism active)")
    print(f"📊 This shows the specific contribution of each optimizer within OptiGradTrust\n")
    
    results = []
    
    for i, optimizer in enumerate(optimizers, 1):
        print(f"\n{'='*80}")
        print(f"Testing {i}/{len(optimizers)}: {optimizer}")
        print(f"{'='*80}")
        
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
    
    baseline_acc = None
    for r in results:
        print(f"{r['optimizer']:<15} {r['final_accuracy']:<12.4f} {r['improvement']:<15.4f} {r['detection_f1']:<12.4f}")
        if r['optimizer'] == 'FedBN-P':
            baseline_acc = r['final_accuracy']
    
    # Show contribution of FedBN-P
    if baseline_acc is not None:
        print(f"\n{'='*80}")
        print(f"📈 FedBN-P CONTRIBUTION ANALYSIS")
        print(f"{'='*80}\n")
        
        for r in results:
            if r['optimizer'] != 'FedBN-P':
                diff = baseline_acc - r['final_accuracy']
                pct = (diff / r['final_accuracy']) * 100 if r['final_accuracy'] > 0 else 0
                print(f"  FedBN-P vs {r['optimizer']:<10}: +{diff:.4f} ({pct:+.2f}%)")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_data = {
        'optimizers': optimizers,
        'results': results,
        'note': 'OptiGradTrust with different optimizers - all using trust mechanism',
        'contribution_analysis': {
            'baseline_optimizer': 'FedBN-P',
            'baseline_accuracy': baseline_acc
        },
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
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   This experiment shows how different optimizers perform")
    print(f"   WITHIN the OptiGradTrust framework (trust mechanism active).")
    print(f"   The difference shows FedBN-P's specific contribution.")
    
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

