"""
Combined Attacks Module
=======================

Tests system robustness against multiple simultaneous attack types.
Addresses Reviewer 2's concern: "Do attacks occur in isolated manner or combined?"

Attack combinations tested:
1. Scaling + Noise
2. Sign Flipping + Label Flipping
3. Partial Scaling + Noise
4. Scaling + Sign Flipping
5. All Combined (worst case)
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


class CombinedAttackClient(Client):
    """Client that can perform multiple attacks simultaneously."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_types = []  # Multiple attacks
    
    def set_combined_attacks(self, attack_types, **params):
        """Set multiple attack types."""
        self.attack_types = attack_types
        self.attack_params = params
        print(f"  Client {self.client_id}: Combined attacks = {attack_types}")
    
    def train(self, global_model):
        """Train and apply all attacks."""
        # Get base gradient
        gradient = super().train(global_model)
        
        if not self.is_malicious or not self.attack_types:
            return gradient
        
        # Apply each attack sequentially
        for attack_type in self.attack_types:
            gradient = self._apply_single_attack(gradient, attack_type)
        
        return gradient
    
    def _apply_single_attack(self, gradient, attack_type):
        """Apply a single attack to gradient."""
        if attack_type == 'scaling_attack':
            scaling_factor = self.attack_params.get('scaling_factor', 10.0)
            return gradient * scaling_factor
        
        elif attack_type == 'noise_attack':
            noise_factor = self.attack_params.get('noise_factor', 0.1)
            noise = torch.randn_like(gradient) * noise_factor * gradient.std()
            return gradient + noise
        
        elif attack_type == 'sign_flipping_attack':
            return -gradient
        
        elif attack_type == 'partial_scaling_attack':
            partial_percent = self.attack_params.get('partial_percent', 0.5)
            scaling_factor = self.attack_params.get('partial_scaling_factor', 5.0)
            mask = torch.rand_like(gradient) < partial_percent
            gradient[mask] *= scaling_factor
            return gradient
        
        # Label flipping is handled in data loading, not gradient
        return gradient


def run_combined_attack_experiment(combination_name, attack_types, output_dir, num_rounds=25):
    """Run experiment with combined attacks."""
    
    print(f"\n{'='*80}")
    print(f"🔥 COMBINED ATTACK: {combination_name}")
    print(f"   Attacks: {attack_types}")
    print(f"{'='*80}\n")
    
    set_random_seeds(42)
    
    # Load dataset
    print("Loading dataset...")
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
        num_clients=NUM_CLIENTS,
        iid=not ENABLE_NON_IID,
        alpha=DIRICHLET_ALPHA if ENABLE_NON_IID else None
    )
    
    # Create clients with combined attack capability
    clients = []
    for i in range(NUM_CLIENTS):
        client = CombinedAttackClient(client_id=i, dataset=client_datasets[i], is_malicious=False)
        clients.append(client)
    
    server.add_clients(clients)
    
    # Configure malicious clients with combined attacks
    num_malicious = int(NUM_CLIENTS * 0.3)
    malicious_indices = np.random.choice(NUM_CLIENTS, num_malicious, replace=False)
    
    print(f"Configuring {num_malicious} malicious clients with combined attacks...")
    
    for i in malicious_indices:
        clients[i].is_malicious = True
        clients[i].set_combined_attacks(
            attack_types=attack_types,
            scaling_factor=10.0,
            noise_factor=0.1,
            partial_scaling_factor=5.0,
            partial_percent=0.5
        )
    
    # Train VAE
    print("Training VAE...")
    root_gradients = server._collect_root_gradients()
    server.vae = server.train_vae(root_gradients, vae_epochs=VAE_EPOCHS)
    
    # Run federated training
    print(f"Running federated learning for {num_rounds} rounds...")
    training_errors, round_metrics = server.train(num_rounds=num_rounds)
    
    # Evaluate results
    final_accuracy = server.evaluate_model()
    improvement = final_accuracy - initial_accuracy
    
    # Extract detection metrics
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
        'combination_name': combination_name,
        'attack_types': attack_types,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1_score,
        'malicious_indices': malicious_indices.tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n✅ Results:")
    print(f"   Final Accuracy: {final_accuracy:.4f}")
    print(f"   Detection F1: {f1_score:.4f}")
    
    return result


def test_combined_attacks(output_dir, combinations=None):
    """
    Test various combinations of attacks.
    
    Args:
        output_dir: Output directory
        combinations: List of attack combinations to test
    
    Returns:
        Dictionary with all results
    """
    
    print(f"\n{'🔥'*40}")
    print(f"COMBINED ATTACKS TEST")
    print(f"{'🔥'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default combinations
    if not combinations:
        combinations = [
            ('Scaling + Noise', ['scaling_attack', 'noise_attack']),
            ('Sign Flip + Label Flip', ['sign_flipping_attack', 'label_flipping']),
            ('Partial Scaling + Noise', ['partial_scaling_attack', 'noise_attack']),
            ('Scaling + Sign Flip', ['scaling_attack', 'sign_flipping_attack']),
            ('All Combined', ['scaling_attack', 'noise_attack', 'sign_flipping_attack', 'partial_scaling_attack'])
        ]
    else:
        # Convert if passed as list of attack types
        combinations = [(', '.join(c), c) for c in combinations]
    
    results = {}
    
    for i, (name, attacks) in enumerate(combinations):
        print(f"\n{'='*80}")
        print(f"Combination {i+1}/{len(combinations)}: {name}")
        print(f"{'='*80}")
        
        result = run_combined_attack_experiment(
            combination_name=name,
            attack_types=attacks,
            output_dir=output_dir
        )
        
        results[name] = result
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 COMBINED ATTACKS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Combination':<30} {'Accuracy':<12} {'Detection F1':<12}")
    print(f"{'-'*80}")
    
    for name, res in results.items():
        print(f"{name:<30} {res['final_accuracy']:<12.4f} {res['detection_f1']:<12.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'combined_attacks_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    import pandas as pd
    
    data = []
    for name, res in results.items():
        data.append({
            'Combination': name,
            'Attack_Types': ', '.join(res['attack_types']),
            'Final_Accuracy': res['final_accuracy'],
            'Improvement': res['improvement'],
            'Detection_Precision': res['detection_precision'],
            'Detection_Recall': res['detection_recall'],
            'Detection_F1': res['detection_f1']
        })
    
    df = pd.DataFrame(data)
    csv_file = output_dir / f'combined_attacks_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"\n✅ Combined attacks test completed!")
    print(f"📁 Results: {results_file}")
    print(f"📊 CSV: {csv_file}")
    
    return {
        'results_file': str(results_file),
        'csv_file': str(csv_file),
        'results': results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='experiments/results/combined_attacks')
    
    args = parser.parse_args()
    
    test_combined_attacks(output_dir=args.output_dir)

