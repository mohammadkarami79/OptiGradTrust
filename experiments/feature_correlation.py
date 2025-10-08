"""
Feature Correlation Analysis
=============================

Analyzes correlation between the 6 fingerprint features to verify independence.
Addresses Reviewer 3's concern about feature redundancy.

Computes:
- Pearson correlation matrix
- Heatmap visualization
- Independence analysis
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds


def analyze_correlation(output_dir, generate_heatmap=True, num_samples=200):
    """Analyze feature correlation."""
    
    print(f"\n{'🔗'*40}")
    print(f"FEATURE CORRELATION ANALYSIS")
    print(f"{'🔗'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_random_seeds(42)
    
    root_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    server = Server()
    server.set_datasets(root_loader, test_dataset)
    server._pretrain_global_model()
    
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
    
    # Train VAE
    root_gradients = server._collect_root_gradients()
    server.vae = server.train_vae(root_gradients, vae_epochs=VAE_EPOCHS)
    
    # Collect feature samples
    print(f"Collecting {num_samples} feature samples...")
    
    all_features = []
    
    for _ in range(num_samples):
        # Random client
        client = np.random.choice(clients)
        
        # Random malicious status
        client.is_malicious = np.random.rand() < 0.3
        if client.is_malicious:
            client.set_attack_parameters(
                attack_type=np.random.choice(['scaling_attack', 'noise_attack', 'sign_flipping_attack']),
                scaling_factor=np.random.uniform(5, 15)
            )
        
        # Get gradient
        gradient = client.train(server.global_model)
        
        # Compute features
        features = server._compute_gradient_features(gradient)
        all_features.append(features.cpu().numpy())
        
        client.is_malicious = False
    
    all_features = np.array(all_features)
    
    # Compute correlation matrix
    feature_names = ['VAE Error', 'Cos(Ref)', 'Cos(Peer)', 'L2 Norm', 'Sign Cons.', 'Shapley']
    
    if all_features.shape[1] < 6:
        feature_names = feature_names[:all_features.shape[1]]
    
    corr_matrix = np.corrcoef(all_features.T)
    
    print(f"\n📊 Correlation Matrix:")
    print(f"{'':>12}", end='')
    for name in feature_names:
        print(f"{name:>12}", end='')
    print()
    
    for i, name in enumerate(feature_names):
        print(f"{name:>12}", end='')
        for j in range(len(feature_names)):
            print(f"{corr_matrix[i,j]:>12.3f}", end='')
        print()
    
    # Heatmap
    if generate_heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    xticklabels=feature_names, yticklabels=feature_names,
                    vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        heatmap_path = output_dir / f'correlation_heatmap_{timestamp}.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📈 Heatmap saved: {heatmap_path}")
    
    # Analysis
    print(f"\n📋 Independence Analysis:")
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.7:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
                print(f"  ⚠️  High correlation: {feature_names[i]} ↔ {feature_names[j]}: {corr_matrix[i,j]:.3f}")
    
    if not high_corr_pairs:
        print(f"  ✅ All features are sufficiently independent (correlation < 0.7)")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'correlation_matrix': corr_matrix.tolist(),
        'feature_names': feature_names,
        'high_correlation_pairs': [{'feature1': p[0], 'feature2': p[1], 'correlation': float(p[2])} 
                                    for p in high_corr_pairs],
        'num_samples': num_samples,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / f'correlation_analysis_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    import pandas as pd
    df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)
    csv_file = output_dir / f'correlation_matrix_{timestamp}.csv'
    df.to_csv(csv_file)
    
    print(f"\n✅ Correlation analysis completed!")
    print(f"📁 Results: {results_file}")
    
    return {
        'results_file': str(results_file),
        'csv_file': str(csv_file),
        'heatmap_path': str(heatmap_path) if generate_heatmap else None,
        'results': results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='experiments/results/correlation')
    parser.add_argument('--samples', type=int, default=200)
    
    args = parser.parse_args()
    
    analyze_correlation(output_dir=args.output_dir, num_samples=args.samples)

