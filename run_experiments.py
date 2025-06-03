#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for Federated Learning System

This script runs multiple experiments with different configurations and saves
all results in an organized manner for easy comparison and analysis.

Usage:
    python run_experiments.py --config experiment_configs.json
    python run_experiments.py --single mnist_cnn_fedbn
    python run_experiments.py --compare
"""

import os
import sys
import json
import argparse
import datetime
import shutil
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from federated_learning.config.config import *

def create_experiment_name(config: Dict[str, Any]) -> str:
    """
    Create a descriptive experiment name based on configuration.
    
    Format: {dataset}_{model}_{aggregation}_{attack}_{timestamp}
    Example: mnist_cnn_fedbn_partial_scaling_20241202_143022
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract key configuration elements
    dataset = config.get('DATASET', 'unknown').lower()
    model = config.get('MODEL', 'unknown').lower()
    aggregation = config.get('AGGREGATION_METHOD', 'unknown').lower()
    attack = config.get('ATTACK_TYPE', 'none').lower()
    
    # Clean up attack name
    attack = attack.replace('_attack', '').replace('_', '')
    
    return f"{dataset}_{model}_{aggregation}_{attack}_{timestamp}"

def save_experiment_config(experiment_name: str, config: Dict[str, Any]):
    """Save the experiment configuration."""
    config_path = os.path.join('results', 'configs', f'{experiment_name}_config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"✓ Configuration saved: {config_path}")

def save_experiment_results(experiment_name: str, results: Dict[str, Any]):
    """Save comprehensive experiment results."""
    
    # Create experiment directory
    exp_dir = os.path.join('results', 'experiments', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save main results
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save metrics summary
    metrics_summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'final_accuracy': results.get('final_accuracy', 0.0),
        'initial_accuracy': results.get('initial_accuracy', 0.0),
        'accuracy_improvement': results.get('final_accuracy', 0.0) - results.get('initial_accuracy', 0.0),
        'detection_metrics': results.get('detection_metrics', {}),
        'training_rounds': len(results.get('test_errors', [])),
        'malicious_detection_rate': results.get('detection_metrics', {}).get('malicious_detection_rate', 0.0),
        'false_positive_rate': results.get('detection_metrics', {}).get('false_positive_rate', 0.0),
        'average_trust_score_honest': results.get('detection_metrics', {}).get('avg_trust_honest', 0.0),
        'average_trust_score_malicious': results.get('detection_metrics', {}).get('avg_trust_malicious', 0.0)
    }
    
    metrics_path = os.path.join('results', 'metrics', f'{experiment_name}_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    
    print(f"✓ Results saved: {results_path}")
    print(f"✓ Metrics saved: {metrics_path}")
    
    return exp_dir

def create_comprehensive_plots(experiment_name: str, results: Dict[str, Any], exp_dir: str):
    """Create comprehensive plots for the experiment."""
    
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get data from results
    test_errors = results.get('test_errors', [])
    round_metrics = results.get('round_metrics', {})
    
    if not test_errors or not round_metrics:
        print("Warning: Insufficient data for plotting")
        return
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Experiment Results: {experiment_name}', fontsize=16, fontweight='bold')
    
    # 1. Test Error Over Time
    axes[0, 0].plot(test_errors, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_title('Test Error Over Rounds')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Error Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Trust Scores by Client Type
    honest_trust_scores = []
    malicious_trust_scores = []
    rounds = []
    
    for round_idx, round_data in round_metrics.items():
        if isinstance(round_idx, str):
            round_idx = int(round_idx)
        rounds.append(round_idx + 1)
        
        trust_scores = round_data.get('trust_scores', {})
        client_status = round_data.get('client_status', {})
        
        round_honest = []
        round_malicious = []
        
        for client_id, trust_score in trust_scores.items():
            if isinstance(client_id, str):
                client_id = int(client_id)
            
            status = client_status.get(client_id, client_status.get(str(client_id), 'UNKNOWN'))
            if status == 'HONEST':
                round_honest.append(trust_score)
            elif status == 'MALICIOUS':
                round_malicious.append(trust_score)
        
        honest_trust_scores.append(np.mean(round_honest) if round_honest else 0)
        malicious_trust_scores.append(np.mean(round_malicious) if round_malicious else 0)
    
    if honest_trust_scores and malicious_trust_scores:
        axes[0, 1].plot(rounds, honest_trust_scores, 'g-', linewidth=2, marker='o', label='Honest Clients')
        axes[0, 1].plot(rounds, malicious_trust_scores, 'r-', linewidth=2, marker='x', label='Malicious Clients')
        axes[0, 1].set_title('Average Trust Scores by Client Type')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Trust Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Aggregation Weights by Client Type
    honest_weights = []
    malicious_weights = []
    
    for round_idx, round_data in round_metrics.items():
        weights = round_data.get('weights', {})
        client_status = round_data.get('client_status', {})
        
        round_honest = []
        round_malicious = []
        
        for client_id, weight in weights.items():
            if isinstance(client_id, str):
                client_id = int(client_id)
            
            status = client_status.get(client_id, client_status.get(str(client_id), 'UNKNOWN'))
            if status == 'HONEST':
                round_honest.append(weight)
            elif status == 'MALICIOUS':
                round_malicious.append(weight)
        
        honest_weights.append(np.mean(round_honest) if round_honest else 0)
        malicious_weights.append(np.mean(round_malicious) if round_malicious else 0)
    
    if honest_weights and malicious_weights:
        axes[0, 2].plot(rounds, honest_weights, 'g-', linewidth=2, marker='o', label='Honest Clients')
        axes[0, 2].plot(rounds, malicious_weights, 'r-', linewidth=2, marker='x', label='Malicious Clients')
        axes[0, 2].set_title('Average Aggregation Weights by Client Type')
        axes[0, 2].set_xlabel('Round')
        axes[0, 2].set_ylabel('Weight')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Gradient Norms by Client Type
    honest_norms = []
    malicious_norms = []
    
    for round_idx, round_data in round_metrics.items():
        raw_metrics = round_data.get('raw_metrics', {})
        
        round_honest = []
        round_malicious = []
        
        for client_id, metrics in raw_metrics.items():
            if isinstance(client_id, str):
                client_id = int(client_id)
            
            is_malicious = metrics.get('is_malicious', False)
            if is_malicious:
                # Use attacked norm for malicious clients
                norm = metrics.get('attacked_norm', metrics.get('original_norm', 0))
                round_malicious.append(norm)
            else:
                norm = metrics.get('original_norm', 0)
                round_honest.append(norm)
        
        honest_norms.append(np.mean(round_honest) if round_honest else 0)
        malicious_norms.append(np.mean(round_malicious) if round_malicious else 0)
    
    if honest_norms and malicious_norms:
        axes[1, 0].plot(rounds, honest_norms, 'g-', linewidth=2, marker='o', label='Honest Clients')
        axes[1, 0].plot(rounds, malicious_norms, 'r-', linewidth=2, marker='x', label='Malicious Clients')
        axes[1, 0].set_title('Average Gradient Norms by Client Type')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')  # Log scale for better visualization
    
    # 5. Detection Performance Metrics
    detection_metrics = results.get('detection_metrics', {})
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    metrics_values = [
        detection_metrics.get('precision', 0),
        detection_metrics.get('recall', 0),
        detection_metrics.get('f1_score', 0),
        detection_metrics.get('accuracy', 0)
    ]
    
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[1, 1].set_title('Malicious Client Detection Performance')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].annotate(f'{value:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
    
    # 6. Feature Distribution Comparison (if available)
    if round_metrics:
        # Get features from the last round
        last_round = max(round_metrics.keys(), key=lambda x: int(x))
        features_data = round_metrics[last_round].get('features', {})
        client_status = round_metrics[last_round].get('client_status', {})
        
        if features_data:
            honest_features = []
            malicious_features = []
            
            for client_id, features in features_data.items():
                if isinstance(client_id, str):
                    client_id = int(client_id)
                
                status = client_status.get(client_id, client_status.get(str(client_id), 'UNKNOWN'))
                if status == 'HONEST':
                    honest_features.append(features)
                elif status == 'MALICIOUS':
                    malicious_features.append(features)
            
            if honest_features and malicious_features:
                honest_features = np.array(honest_features)
                malicious_features = np.array(malicious_features)
                
                feature_names = ['VAE Error', 'Root Sim', 'Client Sim', 'Grad Norm', 'Sign Cons', 'Shapley']
                num_features = min(len(feature_names), honest_features.shape[1])
                
                x = np.arange(num_features)
                width = 0.35
                
                honest_means = np.mean(honest_features[:, :num_features], axis=0)
                malicious_means = np.mean(malicious_features[:, :num_features], axis=0)
                
                axes[1, 2].bar(x - width/2, honest_means, width, label='Honest', color='green', alpha=0.7)
                axes[1, 2].bar(x + width/2, malicious_means, width, label='Malicious', color='red', alpha=0.7)
                
                axes[1, 2].set_title('Feature Comparison (Final Round)')
                axes[1, 2].set_xlabel('Features')
                axes[1, 2].set_ylabel('Average Value')
                axes[1, 2].set_xticks(x)
                axes[1, 2].set_xticklabels(feature_names[:num_features], rotation=45)
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(plots_dir, f'{experiment_name}_comprehensive.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual plots as well
    create_individual_plots(experiment_name, results, plots_dir)
    
    print(f"✓ Comprehensive plot saved: {plot_path}")

def create_individual_plots(experiment_name: str, results: Dict[str, Any], plots_dir: str):
    """Create individual plots for detailed analysis."""
    
    # Trust scores plot
    plt.figure(figsize=(10, 6))
    round_metrics = results.get('round_metrics', {})
    
    for round_idx, round_data in round_metrics.items():
        trust_scores = round_data.get('trust_scores', {})
        client_status = round_data.get('client_status', {})
        
        for client_id, trust_score in trust_scores.items():
            if isinstance(client_id, str):
                client_id = int(client_id)
            
            status = client_status.get(client_id, client_status.get(str(client_id), 'UNKNOWN'))
            color = 'red' if status == 'MALICIOUS' else 'green'
            marker = 'x' if status == 'MALICIOUS' else 'o'
            
            plt.scatter(int(round_idx) + 1, trust_score, color=color, marker=marker, s=50, alpha=0.7)
    
    plt.title('Trust Scores by Client Type Over Time')
    plt.xlabel('Round')
    plt.ylabel('Trust Score')
    plt.grid(True, alpha=0.3)
    
    # Create custom legend
    import matplotlib.patches as mpatches
    honest_patch = mpatches.Patch(color='green', label='Honest Clients')
    malicious_patch = mpatches.Patch(color='red', label='Malicious Clients')
    plt.legend(handles=[honest_patch, malicious_patch])
    
    plt.savefig(os.path.join(plots_dir, f'{experiment_name}_trust_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Aggregation weights plot
    plt.figure(figsize=(10, 6))
    
    for round_idx, round_data in round_metrics.items():
        weights = round_data.get('weights', {})
        client_status = round_data.get('client_status', {})
        
        for client_id, weight in weights.items():
            if isinstance(client_id, str):
                client_id = int(client_id)
            
            status = client_status.get(client_id, client_status.get(str(client_id), 'UNKNOWN'))
            color = 'red' if status == 'MALICIOUS' else 'green'
            marker = 'x' if status == 'MALICIOUS' else 'o'
            
            plt.scatter(int(round_idx) + 1, weight, color=color, marker=marker, s=50, alpha=0.7)
    
    plt.title('Aggregation Weights by Client Type Over Time')
    plt.xlabel('Round')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.3)
    plt.legend(handles=[honest_patch, malicious_patch])
    
    plt.savefig(os.path.join(plots_dir, f'{experiment_name}_weights.png'), dpi=300, bbox_inches='tight')
    plt.close()

def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment with the given configuration."""
    
    # Create experiment name
    experiment_name = create_experiment_name(config)
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Save configuration
    save_experiment_config(experiment_name, config)
    
    # Update global configuration
    for key, value in config.items():
        if hasattr(sys.modules['federated_learning.config.config'], key):
            setattr(sys.modules['federated_learning.config.config'], key, value)
            globals()[key] = value
    
    # Import and run main script
    import main
    
    # Capture results
    # The main script should return results, but for now we'll collect what we can
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'timestamp': datetime.datetime.now().isoformat(),
        'status': 'completed'
    }
    
    # Run the main experiment
    try:
        # This will run the main experiment
        main.main()
        results['status'] = 'completed'
    except Exception as e:
        print(f"Error in experiment: {str(e)}")
        results['status'] = 'failed'
        results['error'] = str(e)
        return results
    
    # Save results
    exp_dir = save_experiment_results(experiment_name, results)
    
    # Create plots
    try:
        create_comprehensive_plots(experiment_name, results, exp_dir)
    except Exception as e:
        print(f"Warning: Failed to create plots: {str(e)}")
    
    return results

def load_experiment_configs(config_file: str = None) -> List[Dict[str, Any]]:
    """Load experiment configurations from file or return default configs."""
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    # Default experiment configurations
    default_configs = [
        {
            'name': 'MNIST_CNN_FedBN_Partial_Scaling',
            'DATASET': 'MNIST',
            'MODEL': 'CNN',
            'AGGREGATION_METHOD': 'fedbn',
            'ATTACK_TYPE': 'partial_scaling_attack',
            'NUM_CLIENTS': 10,
            'FRACTION_MALICIOUS': 0.3,
            'GLOBAL_EPOCHS': 5,
            'SCALING_FACTOR': 20.0,
        },
        {
            'name': 'MNIST_CNN_FedAvg_Sign_Flipping',
            'DATASET': 'MNIST',
            'MODEL': 'CNN',
            'AGGREGATION_METHOD': 'fedavg_with_trust',
            'ATTACK_TYPE': 'sign_flipping_attack',
            'NUM_CLIENTS': 10,
            'FRACTION_MALICIOUS': 0.3,
            'GLOBAL_EPOCHS': 5,
        },
        {
            'name': 'ALZHEIMER_RESNET18_FedBN_Scaling',
            'DATASET': 'ALZHEIMER',
            'MODEL': 'RESNET18',
            'AGGREGATION_METHOD': 'fedbn',
            'ATTACK_TYPE': 'scaling_attack',
            'NUM_CLIENTS': 8,
            'FRACTION_MALICIOUS': 0.25,
            'GLOBAL_EPOCHS': 3,
            'SCALING_FACTOR': 15.0,
        }
    ]
    
    return default_configs

def create_comparison_report(results_dir: str = 'results/metrics'):
    """Create a comprehensive comparison report of all experiments."""
    
    print(f"\n{'='*60}")
    print("Creating Comparison Report")
    print(f"{'='*60}")
    
    # Load all metrics files
    metrics_files = []
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('_metrics.json'):
                metrics_files.append(os.path.join(results_dir, file))
    
    if not metrics_files:
        print("No experiment metrics found for comparison")
        return
    
    # Load all metrics
    all_metrics = []
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not all_metrics:
        print("No valid metrics found")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment Comparison Report', fontsize=16, fontweight='bold')
    
    # Plot 1: Final Accuracy Comparison
    if 'final_accuracy' in df.columns:
        exp_names = [name.split('_')[-1] for name in df['experiment_name']]  # Get timestamp part
        axes[0, 0].bar(range(len(df)), df['final_accuracy'], alpha=0.7)
        axes[0, 0].set_title('Final Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(exp_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Detection Performance
    if 'malicious_detection_rate' in df.columns:
        axes[0, 1].bar(range(len(df)), df['malicious_detection_rate'], alpha=0.7, color='orange')
        axes[0, 1].set_title('Malicious Detection Rate')
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(exp_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Trust Score Differences
    if 'average_trust_score_honest' in df.columns and 'average_trust_score_malicious' in df.columns:
        trust_diff = df['average_trust_score_honest'] - df['average_trust_score_malicious']
        axes[1, 0].bar(range(len(df)), trust_diff, alpha=0.7, color='green')
        axes[1, 0].set_title('Trust Score Difference (Honest - Malicious)')
        axes[1, 0].set_ylabel('Trust Score Difference')
        axes[1, 0].set_xticks(range(len(df)))
        axes[1, 0].set_xticklabels(exp_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Training Rounds vs Performance
    if 'training_rounds' in df.columns and 'final_accuracy' in df.columns:
        axes[1, 1].scatter(df['training_rounds'], df['final_accuracy'], alpha=0.7, s=100)
        axes[1, 1].set_title('Training Rounds vs Final Accuracy')
        axes[1, 1].set_xlabel('Training Rounds')
        axes[1, 1].set_ylabel('Final Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join('results', 'comparison_report.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison CSV
    csv_path = os.path.join('results', 'comparison_summary.csv')
    df.to_csv(csv_path, index=False)
    
    # Create summary table
    print("\nExperiment Comparison Summary:")
    print("="*80)
    for idx, row in df.iterrows():
        print(f"Experiment: {row['experiment_name']}")
        print(f"  Final Accuracy: {row.get('final_accuracy', 'N/A'):.4f}")
        print(f"  Detection Rate: {row.get('malicious_detection_rate', 'N/A'):.4f}")
        print(f"  Trust Score Diff: {row.get('average_trust_score_honest', 0) - row.get('average_trust_score_malicious', 0):.4f}")
        print()
    
    print(f"✓ Comparison report saved: {comparison_path}")
    print(f"✓ Comparison CSV saved: {csv_path}")

def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Run federated learning experiments')
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--single', type=str, help='Run single experiment with given name')
    parser.add_argument('--compare', action='store_true', help='Create comparison report')
    parser.add_argument('--list', action='store_true', help='List available experiment configurations')
    
    args = parser.parse_args()
    
    if args.compare:
        create_comparison_report()
        return
    
    # Load configurations
    configs = load_experiment_configs(args.config)
    
    if args.list:
        print("Available experiment configurations:")
        for i, config in enumerate(configs):
            print(f"{i+1}. {config.get('name', f'Config {i+1}')}")
        return
    
    if args.single:
        # Find specific configuration by name
        target_config = None
        for config in configs:
            if args.single.lower() in config.get('name', '').lower():
                target_config = config
                break
        
        if target_config:
            configs = [target_config]
        else:
            print(f"Configuration '{args.single}' not found")
            return
    
    # Run experiments
    print(f"Running {len(configs)} experiment(s)...")
    
    all_results = []
    for i, config in enumerate(configs):
        print(f"\nStarting experiment {i+1}/{len(configs)}")
        try:
            results = run_single_experiment(config)
            all_results.append(results)
        except Exception as e:
            print(f"Failed to run experiment {i+1}: {str(e)}")
    
    # Create final comparison if multiple experiments
    if len(all_results) > 1:
        print(f"\nCreating final comparison report...")
        create_comparison_report()
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved in: results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 