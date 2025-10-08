"""
Extended Metrics Module
========================

Computes comprehensive evaluation metrics beyond accuracy:
1. Precision, Recall, F1-Score (per class and macro/micro)
2. AUC-ROC
3. Confusion Matrix
4. Per-class performance

Addresses Reviewers 1 & 2's request for more evaluation metrics.
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds


def compute_all_metrics(model, test_dataset, device, num_classes):
    """Compute all classification metrics."""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Predictions
            pred = output.argmax(dim=1, keepdim=False)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # Probabilities (for AUC)
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 1. Precision, Recall, F1 (per class and averaged)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Macro and micro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )
    
    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 3. AUC-ROC (multi-class)
    try:
        if num_classes == 2:
            # Binary classification
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # Multi-class (one-vs-rest)
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"  Warning: Could not compute AUC: {e}")
        auc_score = 0.0
    
    # 4. Per-class metrics
    per_class_metrics = []
    for i in range(num_classes):
        per_class_metrics.append({
            'class': i,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        })
    
    # 5. Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'precision_micro': float(precision_micro),
        'recall_macro': float(recall_macro),
        'recall_micro': float(recall_micro),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'auc_roc': float(auc_score),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics
    }
    
    return metrics, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix heatmap."""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Confusion matrix saved: {output_path}")


def plot_per_class_metrics(per_class_metrics, output_path):
    """Plot per-class precision, recall, F1."""
    
    classes = [m['class'] for m in per_class_metrics]
    precision = [m['precision'] for m in per_class_metrics]
    recall = [m['recall'] for m in per_class_metrics]
    f1 = [m['f1'] for m in per_class_metrics]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {c}' for c in classes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Per-class metrics plot saved: {output_path}")


def compute_extended_metrics(output_dir, metrics=['precision', 'recall', 'f1', 'auc', 'confusion_matrix'], num_rounds=25):
    """
    Compute extended evaluation metrics for OptiGradTrust.
    
    Args:
        output_dir: Output directory
        metrics: List of metrics to compute
        num_rounds: Number of training rounds
    
    Returns:
        Dictionary with all metrics
    """
    
    print(f"\n{'📊'*40}")
    print(f"EXTENDED METRICS COMPUTATION")
    print(f"{'📊'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_random_seeds(42)
    
    # Load dataset
    print("Loading dataset...")
    root_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Determine number of classes
    if DATASET == 'MNIST' or DATASET == 'CIFAR10':
        num_classes = 10
        class_names = [str(i) for i in range(10)]
    elif DATASET == 'ALZHEIMER':
        num_classes = ALZHEIMER_CLASSES
        class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    else:
        num_classes = 10
        class_names = [str(i) for i in range(10)]
    
    print(f"Dataset: {DATASET}, Classes: {num_classes}")
    
    # Create server
    server = Server()
    server.set_datasets(root_loader, test_dataset)
    server._pretrain_global_model()
    
    # Initial metrics
    print("\nComputing INITIAL metrics (before training)...")
    initial_metrics, _, _, _ = compute_all_metrics(
        server.global_model, test_dataset, server.device, num_classes
    )
    
    print(f"  Initial Accuracy: {initial_metrics['accuracy']:.4f}")
    print(f"  Initial F1 (macro): {initial_metrics['f1_macro']:.4f}")
    print(f"  Initial AUC: {initial_metrics['auc_roc']:.4f}")
    
    # Create clients
    print("\nCreating clients...")
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
    print("Training VAE...")
    root_gradients = server._collect_root_gradients()
    server.vae = server.train_vae(root_gradients, vae_epochs=VAE_EPOCHS)
    
    # Train model
    print(f"\nRunning federated learning for {num_rounds} rounds...")
    training_errors, round_metrics = server.train(num_rounds=num_rounds)
    
    # Final metrics
    print("\nComputing FINAL metrics (after training)...")
    final_metrics, final_labels, final_preds, final_probs = compute_all_metrics(
        server.global_model, test_dataset, server.device, num_classes
    )
    
    print(f"\n{'='*80}")
    print(f"📈 METRICS COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<25} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
    print(f"{'-'*80}")
    print(f"{'Accuracy':<25} {initial_metrics['accuracy']:<12.4f} {final_metrics['accuracy']:<12.4f} "
          f"{final_metrics['accuracy'] - initial_metrics['accuracy']:<12.4f}")
    print(f"{'Precision (macro)':<25} {initial_metrics['precision_macro']:<12.4f} {final_metrics['precision_macro']:<12.4f} "
          f"{final_metrics['precision_macro'] - initial_metrics['precision_macro']:<12.4f}")
    print(f"{'Recall (macro)':<25} {initial_metrics['recall_macro']:<12.4f} {final_metrics['recall_macro']:<12.4f} "
          f"{final_metrics['recall_macro'] - initial_metrics['recall_macro']:<12.4f}")
    print(f"{'F1 (macro)':<25} {initial_metrics['f1_macro']:<12.4f} {final_metrics['f1_macro']:<12.4f} "
          f"{final_metrics['f1_macro'] - initial_metrics['f1_macro']:<12.4f}")
    print(f"{'AUC-ROC':<25} {initial_metrics['auc_roc']:<12.4f} {final_metrics['auc_roc']:<12.4f} "
          f"{final_metrics['auc_roc'] - initial_metrics['auc_roc']:<12.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for m in final_metrics['per_class_metrics']:
        print(f"  Class {m['class']}: {m['f1']:.4f} (support: {m['support']})")
    
    # Create visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot confusion matrix
    cm_path = output_dir / f'confusion_matrix_{timestamp}.png'
    plot_confusion_matrix(
        np.array(final_metrics['confusion_matrix']),
        class_names,
        cm_path
    )
    
    # Plot per-class metrics
    per_class_path = output_dir / f'per_class_metrics_{timestamp}.png'
    plot_per_class_metrics(final_metrics['per_class_metrics'], per_class_path)
    
    # Save results
    results = {
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics,
        'improvement': {
            'accuracy': final_metrics['accuracy'] - initial_metrics['accuracy'],
            'precision_macro': final_metrics['precision_macro'] - initial_metrics['precision_macro'],
            'recall_macro': final_metrics['recall_macro'] - initial_metrics['recall_macro'],
            'f1_macro': final_metrics['f1_macro'] - initial_metrics['f1_macro'],
            'auc_roc': final_metrics['auc_roc'] - initial_metrics['auc_roc']
        },
        'num_classes': num_classes,
        'class_names': class_names,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_dir / f'extended_metrics_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    import pandas as pd
    
    # Summary CSV
    summary_data = [{
        'Metric': metric,
        'Initial': initial_metrics.get(metric.lower().replace(' ', '_'), 0),
        'Final': final_metrics.get(metric.lower().replace(' ', '_'), 0),
        'Improvement': final_metrics.get(metric.lower().replace(' ', '_'), 0) - 
                      initial_metrics.get(metric.lower().replace(' ', '_'), 0)
    } for metric in ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 'AUC_ROC']]
    
    df_summary = pd.DataFrame(summary_data)
    summary_csv = output_dir / f'metrics_summary_{timestamp}.csv'
    df_summary.to_csv(summary_csv, index=False)
    
    # Per-class CSV
    df_per_class = pd.DataFrame(final_metrics['per_class_metrics'])
    per_class_csv = output_dir / f'per_class_metrics_{timestamp}.csv'
    df_per_class.to_csv(per_class_csv, index=False)
    
    print(f"\n✅ Extended metrics computation completed!")
    print(f"📁 Results: {results_file}")
    print(f"📊 Summary CSV: {summary_csv}")
    print(f"📊 Per-class CSV: {per_class_csv}")
    print(f"📈 Confusion matrix: {cm_path}")
    print(f"📈 Per-class plot: {per_class_path}")
    
    return {
        'results_file': str(results_file),
        'summary_csv': str(summary_csv),
        'per_class_csv': str(per_class_csv),
        'confusion_matrix_plot': str(cm_path),
        'per_class_plot': str(per_class_path),
        'results': results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='experiments/results/extended_metrics')
    parser.add_argument('--rounds', type=int, default=25)
    
    args = parser.parse_args()
    
    compute_extended_metrics(output_dir=args.output_dir, num_rounds=args.rounds)

