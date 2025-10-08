"""
Ablation Study Module
=====================

Tests the contribution of each feature in the 6-dimensional fingerprint by:
1. Training with all features (baseline)
2. Training with each feature removed (drop-one-feature)
3. Comparing performance degradation

Features tested:
1. VAE reconstruction error
2. Cosine similarity to reference
3. Cosine similarity to peers
4. L2 norm
5. Sign consistency
6. Shapley value
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds


class AblationServer(Server):
    """Modified server for ablation studies - can disable specific features."""
    
    def __init__(self, disabled_features=None):
        """
        Args:
            disabled_features: List of feature names to disable
                ['vae', 'cosine_ref', 'cosine_peer', 'l2_norm', 'sign_consistency', 'shapley']
        """
        super().__init__()
        self.disabled_features = disabled_features or []
        print(f"🔬 Ablation Mode: Disabled features = {self.disabled_features}")
    
    def _compute_gradient_features(self, gradient, root_gradient=None, skip_client_sim=False):
        """Override to disable specific features."""
        # Call original implementation with all parameters
        features = super()._compute_gradient_features(gradient, root_gradient, skip_client_sim)
        
        # Feature indices: [vae, cos_ref, cos_peer, l2, sign, shapley]
        feature_map = {
            'vae': 0,
            'cosine_ref': 1,
            'cosine_peer': 2,
            'l2_norm': 3,
            'sign_consistency': 4,
            'shapley': 5
        }
        
        # Zero out disabled features
        for feat_name in self.disabled_features:
            if feat_name in feature_map:
                idx = feature_map[feat_name]
                if idx < len(features):
                    # Set to neutral value (mean or zero depending on feature)
                    if feat_name in ['vae', 'l2_norm']:
                        features[idx] = 0.0  # Zero for non-similarity metrics
                    else:
                        features[idx] = 0.5  # Neutral for similarity metrics
        
        return features


def run_single_ablation_experiment(disabled_feature, output_dir, num_rounds=25, quick_mode=False):
    """Run experiment with one feature disabled."""
    
    print(f"\n{'='*80}")
    print(f"🧪 ABLATION EXPERIMENT: Disabling '{disabled_feature}'")
    print(f"{'='*80}\n")
    
    if quick_mode:
        num_rounds = 10
        print("⚡ Quick mode: Reduced to 10 rounds")
    
    # Set seed for reproducibility
    set_random_seeds(42)
    
    # Load dataset
    print("Loading dataset...")
    root_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Create server with feature disabled
    server = AblationServer(disabled_features=[disabled_feature] if disabled_feature != 'baseline' else [])
    server.set_datasets(root_loader, test_dataset)
    server._pretrain_global_model()
    
    initial_accuracy = server.evaluate_model()
    print(f"Initial accuracy: {initial_accuracy:.4f}")
    
    # Create client datasets
    print("Creating client datasets...")
    root_client_dataset, client_datasets = create_client_datasets(
        train_dataset=root_dataset,
        num_clients=NUM_CLIENTS,
        iid=not ENABLE_NON_IID,
        alpha=DIRICHLET_ALPHA if ENABLE_NON_IID else None
    )
    
    # Create clients
    clients = []
    for i in range(NUM_CLIENTS):
        client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
        clients.append(client)
    
    server.add_clients(clients)
    
    # Configure malicious clients (30% with scaling attack as default)
    num_malicious = int(NUM_CLIENTS * 0.3)
    malicious_indices = np.random.choice(NUM_CLIENTS, num_malicious, replace=False)
    
    print(f"Malicious clients: {malicious_indices}")
    
    for i in malicious_indices:
        clients[i].is_malicious = True
        clients[i].set_attack_parameters(
            attack_type='scaling_attack',
            scaling_factor=10.0
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
        'disabled_feature': disabled_feature,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1_score,
        'malicious_indices': malicious_indices.tolist(),
        'num_rounds': num_rounds,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n✅ Results:")
    print(f"   Final Accuracy: {final_accuracy:.4f}")
    print(f"   Improvement: {improvement:.4f}")
    print(f"   Detection F1: {f1_score:.4f}")
    
    return result


def run_ablation_analysis(output_dir, features_to_test=None, quick_mode=False):
    """
    Run complete ablation study.
    
    Args:
        output_dir: Directory to save results
        features_to_test: List of features to test (None = all features)
        quick_mode: If True, run fewer rounds for quick testing
    
    Returns:
        Dictionary with all ablation results
    """
    
    print(f"\n{'🔬'*40}")
    print(f"ABLATION STUDY - DROP-ONE-FEATURE ANALYSIS")
    print(f"{'🔬'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all features
    all_features = ['vae', 'cosine_ref', 'cosine_peer', 'l2_norm', 'sign_consistency', 'shapley']
    
    if features_to_test:
        features = features_to_test
    else:
        features = all_features
    
    print(f"Testing features: {features}")
    print(f"Quick mode: {quick_mode}\n")
    
    results = {}
    
    # 1. Baseline (all features enabled)
    print("\n" + "="*80)
    print("1. BASELINE: All features enabled")
    print("="*80)
    
    baseline_result = run_single_ablation_experiment(
        disabled_feature='baseline',
        output_dir=output_dir,
        quick_mode=quick_mode
    )
    results['baseline'] = baseline_result
    
    # 2. Test each feature removal
    for i, feature in enumerate(features):
        print(f"\n" + "="*80)
        print(f"{i+2}. DROP FEATURE: {feature}")
        print("="*80)
        
        result = run_single_ablation_experiment(
            disabled_feature=feature,
            output_dir=output_dir,
            quick_mode=quick_mode
        )
        results[f'without_{feature}'] = result
    
    # 3. Analyze results
    print(f"\n{'='*80}")
    print(f"📊 ABLATION ANALYSIS SUMMARY")
    print(f"{'='*80}\n")
    
    baseline_acc = results['baseline']['final_accuracy']
    baseline_f1 = results['baseline']['detection_f1']
    
    analysis = {
        'baseline': {
            'accuracy': baseline_acc,
            'detection_f1': baseline_f1
        },
        'feature_importance': {}
    }
    
    print(f"Baseline Performance:")
    print(f"  Accuracy: {baseline_acc:.4f}")
    print(f"  Detection F1: {baseline_f1:.4f}\n")
    
    print(f"Feature Ablation Results:")
    print(f"{'Feature':<20} {'Accuracy':<12} {'Acc Drop':<12} {'Det F1':<12} {'F1 Drop':<12}")
    print(f"{'-'*80}")
    
    for feature in features:
        key = f'without_{feature}'
        if key in results:
            res = results[key]
            acc_drop = baseline_acc - res['final_accuracy']
            f1_drop = baseline_f1 - res['detection_f1']
            
            print(f"{feature:<20} {res['final_accuracy']:<12.4f} {acc_drop:<12.4f} "
                  f"{res['detection_f1']:<12.4f} {f1_drop:<12.4f}")
            
            analysis['feature_importance'][feature] = {
                'accuracy_drop': acc_drop,
                'f1_drop': f1_drop,
                'final_accuracy': res['final_accuracy'],
                'final_f1': res['detection_f1']
            }
    
    # 4. Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = output_dir / f'ablation_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({'raw_results': results, 'analysis': analysis}, f, indent=2)
    
    # Save summary CSV
    import pandas as pd
    
    summary_data = []
    summary_data.append({
        'Configuration': 'Baseline (All Features)',
        'Accuracy': baseline_acc,
        'Detection_F1': baseline_f1,
        'Accuracy_Drop': 0.0,
        'F1_Drop': 0.0
    })
    
    for feature in features:
        key = f'without_{feature}'
        if key in results:
            res = results[key]
            summary_data.append({
                'Configuration': f'Without {feature}',
                'Accuracy': res['final_accuracy'],
                'Detection_F1': res['detection_f1'],
                'Accuracy_Drop': baseline_acc - res['final_accuracy'],
                'F1_Drop': baseline_f1 - res['detection_f1']
            })
    
    df = pd.DataFrame(summary_data)
    csv_file = output_dir / f'ablation_summary_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"\n✅ Ablation study completed!")
    print(f"📁 Results: {results_file}")
    print(f"📊 Summary: {csv_file}")
    
    return {
        'results_file': str(results_file),
        'csv_file': str(csv_file),
        'analysis': analysis,
        'raw_results': results
    }


if __name__ == "__main__":
    # Test ablation study
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='experiments/results/ablation', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--features', nargs='+', help='Specific features to test')
    
    args = parser.parse_args()
    
    run_ablation_analysis(
        output_dir=args.output_dir,
        features_to_test=args.features,
        quick_mode=args.quick
    )

