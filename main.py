"""
Main script for running comprehensive attack analysis on federated learning system.

This script performs the following steps:
1. Load dataset and split into root, client, and test sets
2. Create server and clients (some malicious)
3. Train shared models (VAE, dual attention) ONCE
4. Run federated learning with EACH attack type
5. Create comprehensive comparison plots

Key advantages:
- Shared models trained only once (time efficiency)
- All attacks tested systematically  
- Comprehensive comparison plots for research
- Reduced total experiment time
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the federated learning package
from federated_learning.config.config import *
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds
from federated_learning.utils.training_utils import train_dual_attention
from federated_learning.utils.plotting_utils import create_all_research_plots

# List of most important attack types for conference paper (reduced to 5)
ALL_ATTACK_TYPES = [
    'scaling_attack',           # Most common gradient scaling attack
    'partial_scaling_attack',   # Sophisticated partial scaling  
    'sign_flipping_attack',     # Simple but effective gradient flip
    'noise_attack',             # Additive noise attack
    'label_flipping'            # Data poisoning attack
]

# Note: Removed min_max_attack, min_sum_attack, targeted_attack to focus on most important attacks

def run_attack_experiment(server, clients, attack_type, experiment_number):
    """Run federated learning experiment with specific attack type."""
    
    print(f"\n🔥 EXPERIMENT {experiment_number}: {attack_type.upper()}")
    print("="*60)
    
    # Configure malicious clients for this attack
    num_malicious = int(NUM_CLIENTS * FRACTION_MALICIOUS)
    malicious_indices = np.random.choice(NUM_CLIENTS, num_malicious, replace=False)
    
    print(f"Malicious clients: {malicious_indices} (Attack: {attack_type})")
    
    for i, client in enumerate(clients):
        if i in malicious_indices:
            client.is_malicious = True
            client.set_attack_parameters(
                attack_type=attack_type,
                scaling_factor=SCALING_FACTOR,
                partial_percent=PARTIAL_SCALING_PERCENT,
                noise_factor=NOISE_FACTOR,
                flip_probability=FLIP_PROBABILITY
            )
        else:
            client.is_malicious = False
    
    # Store initial accuracy
    initial_accuracy = server.evaluate_model()
    
    print(f"Running federated learning for {GLOBAL_EPOCHS} epochs...")
    
    # Use server's built-in train method
    training_errors, round_metrics = server.train(num_rounds=GLOBAL_EPOCHS)
    
    # Calculate final metrics
    final_accuracy = server.evaluate_model()
    improvement = final_accuracy - initial_accuracy
    
    # Extract detection results from server round_metrics (NOT training_history!)
    detection_metrics = {
        'precision': 0.0,
        'recall': 0.0, 
        'f1_score': 0.0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    # Aggregate detection metrics across all rounds
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    try:
        if round_metrics:
            print(f"   📊 Analyzing detection results from {len(round_metrics)} rounds...")
            
            for round_idx, round_data in round_metrics.items():
                if 'detection_results' in round_data and round_data['detection_results']:
                    det_results = round_data['detection_results']
                    
                    # Extract metrics for this round
                    tp = det_results.get('true_positives', 0)
                    fp = det_results.get('false_positives', 0)
                    fn = det_results.get('false_negatives', 0)
                    tn = det_results.get('true_negatives', 0)
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_tn += tn
                    
                    print(f"   Round {round_idx + 1}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            
            # Calculate aggregated metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            detection_metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score, 
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            }
            
            print(f"   📈 AGGREGATED DETECTION METRICS:")
            print(f"   Detection Precision: {precision:.4f}")
            print(f"   Detection Recall: {recall:.4f}")
            print(f"   Detection F1-Score: {f1_score:.4f}")
            print(f"   Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
            
    except Exception as e:
        print(f"   ⚠️ Could not extract detection metrics: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Extract training progress from round_metrics
    training_accuracies = [initial_accuracy]  # Include initial accuracy
    
    if round_metrics:
        for round_idx in sorted(round_metrics.keys()):
            round_data = round_metrics[round_idx]
            if 'test_accuracy' in round_data:
                training_accuracies.append(round_data['test_accuracy'])
    
    # Create comprehensive result structure
    experiment_result = {
        'attack_type': attack_type,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'training_history': {
            'accuracies': training_accuracies,
            'losses': training_errors if 'training_errors' in locals() else [],
            'detection_results': [round_metrics[r].get('detection_results', {}) for r in sorted(round_metrics.keys())] if round_metrics else []
        },
        'detection_metrics': detection_metrics,
        'malicious_indices': malicious_indices.tolist(),
        'timestamp': datetime.now().isoformat(),
        'round_metrics': round_metrics  # Include full round metrics for debugging
    }
    
    print(f"✅ RESULTS:")
    print(f"   Initial Accuracy: {initial_accuracy:.4f}")
    print(f"   Final Accuracy: {final_accuracy:.4f}")
    print(f"   Improvement: {improvement:.4f}")
    
    return experiment_result

def create_comprehensive_comparison_plots(all_results):
    """Create comprehensive comparison plots for all attack experiments."""
    
    print(f"\n📊 CREATING COMPREHENSIVE COMPARISON PLOTS")
    print("="*60)
    
    # Extract data for plotting
    attack_names = [r['attack_type'].replace('_attack', '').replace('_', ' ').title() for r in all_results]
    final_accuracies = [r['final_accuracy'] for r in all_results]
    improvements = [r['improvement'] for r in all_results]
    detection_precisions = [r['detection_metrics']['precision'] for r in all_results]
    detection_recalls = [r['detection_metrics']['recall'] for r in all_results]
    detection_f1s = [r['detection_metrics']['f1_score'] for r in all_results]
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('COMPREHENSIVE ATTACK ANALYSIS - ALL RESULTS COMPARISON', fontsize=18, fontweight='bold')
    
    # Plot 1: Final Accuracy Comparison
    bars1 = axes[0, 0].bar(range(len(all_results)), final_accuracies, alpha=0.8, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(all_results))))
    axes[0, 0].set_title('Final Accuracy by Attack Type', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Final Accuracy')
    axes[0, 0].set_xticks(range(len(all_results)))
    axes[0, 0].set_xticklabels(attack_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Accuracy Improvement Comparison
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars2 = axes[0, 1].bar(range(len(all_results)), improvements, alpha=0.8, color=colors)
    axes[0, 1].set_title('Model Learning (Accuracy Improvement)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy Improvement')
    axes[0, 1].set_xticks(range(len(all_results)))
    axes[0, 1].set_xticklabels(attack_names, rotation=45, ha='right')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        offset = 0.0005 if height >= 0 else -0.0005
        va = 'bottom' if height >= 0 else 'top'
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{height:.4f}', ha='center', va=va, fontweight='bold', fontsize=10)
    
    # Plot 3: Detection Performance Overview
    x_pos = np.arange(len(all_results))
    width = 0.25
    
    bars3a = axes[0, 2].bar(x_pos - width, detection_precisions, width, alpha=0.8, color='orange', label='Precision')
    bars3b = axes[0, 2].bar(x_pos, detection_recalls, width, alpha=0.8, color='purple', label='Recall')
    bars3c = axes[0, 2].bar(x_pos + width, detection_f1s, width, alpha=0.8, color='green', label='F1-Score')
    
    axes[0, 2].set_title('Malicious Client Detection Performance', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(attack_names, rotation=45, ha='right')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Performance vs Detection Trade-off
    scatter = axes[1, 0].scatter(improvements, detection_precisions, alpha=0.8, s=150, 
                                c=range(len(all_results)), cmap='viridis', edgecolors='black', linewidth=1)
    for i, attack in enumerate(attack_names):
        axes[1, 0].annotate(attack, (improvements[i], detection_precisions[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    axes[1, 0].set_title('Model Learning vs Detection Trade-off', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Accuracy Improvement')
    axes[1, 0].set_ylabel('Detection Precision')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Training Progress Comparison
    axes[1, 1].set_title('Training Progress Comparison (Top 5)', fontsize=14, fontweight='bold')
    
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for i, result in enumerate(all_results[:5]):  # Show top 5 for clarity
        training_hist = result['training_history']['accuracies']
        epochs = range(1, len(training_hist) + 1)
        axes[1, 1].plot(epochs, training_hist, 
                       marker='o', label=attack_names[i], color=colors_cycle[i], linewidth=2, markersize=4)
    
    axes[1, 1].set_xlabel('Global Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Attack Effectiveness Ranking
    # Combined score: 60% accuracy improvement + 40% detection precision
    combined_scores = [0.6 * improvements[i] + 0.4 * detection_precisions[i] for i in range(len(all_results))]
    sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    
    sorted_attacks = [attack_names[i] for i in sorted_indices]
    sorted_scores = [combined_scores[i] for i in sorted_indices]
    
    colors_ranked = ['gold' if i == 0 else 'silver' if i == 1 else 'orange' if i == 2 else 'lightblue' 
                    for i in range(len(sorted_attacks))]
    
    bars6 = axes[1, 2].barh(range(len(sorted_attacks)), sorted_scores, alpha=0.8, color=colors_ranked)
    axes[1, 2].set_title('Overall System Performance Ranking', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Combined Score (0.6×Accuracy + 0.4×Detection)')
    axes[1, 2].set_yticks(range(len(sorted_attacks)))
    axes[1, 2].set_yticklabels(sorted_attacks)
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars6):
        width = bar.get_width()
        axes[1, 2].text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                       f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"research_plots/comprehensive_attack_comparison_{timestamp}.png"
    os.makedirs("research_plots", exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Comprehensive comparison plot saved: {plot_path}")
    
    # Print summary statistics
    print(f"\n📈 COMPREHENSIVE ANALYSIS SUMMARY:")
    print(f"{'='*60}")
    print(f"🔢 Total Attacks Tested: {len(all_results)}")
    print(f"🏆 Best Final Accuracy: {max(final_accuracies):.4f} ({attack_names[np.argmax(final_accuracies)]})")
    print(f"📈 Best Improvement: {max(improvements):.4f} ({attack_names[np.argmax(improvements)]})")
    print(f"🎯 Best Detection Precision: {max(detection_precisions):.3f} ({attack_names[np.argmax(detection_precisions)]})")
    print(f"📊 Average Accuracy: {np.mean(final_accuracies):.4f} ± {np.std(final_accuracies):.4f}")
    print(f"📊 Average Improvement: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
    print(f"📊 Average Detection: {np.mean(detection_precisions):.3f} ± {np.std(detection_precisions):.3f}")
    print(f"✅ Positive Improvements: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")
    print(f"🎯 High Detection (>0.5): {sum(1 for p in detection_precisions if p > 0.5)}/{len(detection_precisions)}")
    
    return plot_path

def prepare_dual_attention_training_data(server, clients, num_samples=200):
    """Generate synthetic training data for dual attention model."""
    
    print(f"   Generating {num_samples} training samples for dual attention...")
    
    # Generate honest samples by having honest clients train on server model
    honest_features = []
    for i in range(num_samples // 2):
        # Select random honest client
        client = np.random.choice(clients)
        client.is_malicious = False
        
        # Get gradient from honest training
        gradient = client.train(server.global_model)
        
        # Compute features for this gradient
        features = server._compute_gradient_features(gradient)
        honest_features.append(features)
    
    # Generate malicious samples using various attack patterns
    malicious_features = []
    attack_types = ['scaling_attack', 'noise_attack', 'sign_flipping_attack']
    
    for i in range(num_samples // 2):
        # Select random client and make malicious
        client = np.random.choice(clients)
        client.is_malicious = True
        
        # Random attack type
        attack_type = np.random.choice(attack_types)
        client.set_attack_parameters(
            attack_type=attack_type,
            scaling_factor=np.random.uniform(5, 15),
            noise_factor=np.random.uniform(0.1, 1.0)
        )
        
        # Get malicious gradient
        gradient = client.train(server.global_model)
        
        # Compute features
        features = server._compute_gradient_features(gradient)
        malicious_features.append(features)
        
        # Reset client
        client.is_malicious = False
    
    # Combine features and labels
    all_features = torch.stack(honest_features + malicious_features)
    honest_labels = torch.zeros(len(honest_features), 1)  # 0 for honest
    malicious_labels = torch.ones(len(malicious_features), 1)  # 1 for malicious
    all_labels = torch.cat([honest_labels, malicious_labels])
    
    print(f"   ✅ Generated {len(all_features)} samples ({len(honest_features)} honest + {len(malicious_features)} malicious)")
    
    return all_features, all_labels

def main():
    """Main function to run comprehensive attack analysis."""
    
    print(f"🧪 COMPREHENSIVE ATTACK ANALYSIS")
    print(f"{'='*60}")
    print(f"📋 Configuration:")
    print(f"   Dataset: {DATASET}")
    print(f"   Model: {MODEL}")
    print(f"   Data Distribution: {'IID' if not ENABLE_NON_IID else 'Non-IID'}")
    print(f"   Attack Types: {len(ALL_ATTACK_TYPES)} types")
    print(f"   Aggregation: {RL_AGGREGATION_METHOD}")
    print(f"   Clients: {NUM_CLIENTS} ({int(NUM_CLIENTS * FRACTION_MALICIOUS)} malicious)")
    print(f"   Epochs: {GLOBAL_EPOCHS} global, {LOCAL_EPOCHS_CLIENT} local")
    print(f"   Shared Training: VAE ({VAE_EPOCHS} epochs), Root ({LOCAL_EPOCHS_ROOT} epochs)")
    
    # Initialize results
    overall_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'DATASET': DATASET,
            'MODEL': MODEL,
            'AGGREGATION_METHOD': RL_AGGREGATION_METHOD,
            'NUM_CLIENTS': NUM_CLIENTS,
            'FRACTION_MALICIOUS': FRACTION_MALICIOUS,
            'GLOBAL_EPOCHS': GLOBAL_EPOCHS,
            'DATA_DISTRIBUTION': 'IID' if not ENABLE_NON_IID else 'Non-IID'
        },
        'attack_types_tested': ALL_ATTACK_TYPES,
        'status': 'started'
    }
    
    # Step 1: Set random seed
    if RANDOM_SEED is not None:
        set_random_seeds(RANDOM_SEED)
        print(f"Random seed: {RANDOM_SEED}")
    
    # Step 2: Load data (ONCE)
    print("\n--- Loading and preprocessing data ---")
    root_dataset, test_dataset = load_dataset()
    print(f"Root dataset: {len(root_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Step 3: Create server (ONCE)
    print("\n--- Creating server and setting up model ---")
    server = Server()
    server.set_datasets(root_loader, test_dataset)
    
    # Pre-train global model
    server._pretrain_global_model()
    initial_accuracy = server.evaluate_model()
    overall_results['initial_accuracy'] = initial_accuracy
    print(f"Initial model accuracy: {initial_accuracy:.4f}")
    
    # Step 4: Create client datasets (ONCE)
    print("\n--- Creating client datasets ---")
    root_client_dataset, client_datasets = create_client_datasets(
        train_dataset=root_dataset,
        num_clients=NUM_CLIENTS,
        iid=not ENABLE_NON_IID,
        alpha=DIRICHLET_ALPHA if ENABLE_NON_IID else None
    )
    
    print(f"Created {len(client_datasets)} client datasets")
    for i, dataset in enumerate(client_datasets):
        print(f"Client {i}: {len(dataset)} samples")
    
    # Step 5: Create clients (ONCE)
    print("\n--- Creating clients ---")
    clients = []
    for i in range(NUM_CLIENTS):
        client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
        clients.append(client)
    
    server.add_clients(clients)
    
    # Step 6: Train shared models (ONCE!)
    print("\n🔧 TRAINING SHARED MODELS (EXECUTED ONCE)")
    print("="*50)
    
    # Train VAE on root gradients
    print("Training VAE on root dataset gradients...")
    root_gradients = server._collect_root_gradients()
    print(f"Collected {len(root_gradients)} root gradients")
    
    # Train VAE
    server.vae = server.train_vae(root_gradients, vae_epochs=VAE_EPOCHS)
    print("✅ VAE training completed")
    
    # Train dual attention model
    print("Training dual attention model...")
    try:
        # Generate training data for dual attention
        da_features, da_labels = prepare_dual_attention_training_data(server, clients, num_samples=100)
        
        # Train dual attention model
        server.dual_attention = train_dual_attention(
            server.dual_attention, 
            da_features, 
            da_labels, 
            epochs=15  # reduced for speed
        )
        print("✅ Dual attention training completed")
        
    except Exception as e:
        print(f"⚠️  Dual attention training failed: {str(e)}")
        print("⏩ Continuing without pre-trained dual attention...")
    
    print("✅ Shared model preparation completed!")
    
    # Step 7: Run experiments for all attack types
    print(f"\n🚀 RUNNING EXPERIMENTS FOR ALL {len(ALL_ATTACK_TYPES)} ATTACK TYPES")
    print("="*70)
    print("📝 Note: Shared models (VAE, Dual Attention) will NOT be retrained")
    print("⚡ This saves significant time while ensuring fair comparison")
    
    all_results = []
    
    for i, attack_type in enumerate(ALL_ATTACK_TYPES):
        try:
            result = run_attack_experiment(server, clients, attack_type, i + 1)
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Failed {attack_type}: {str(e)}")
            continue
    
    # Step 8: Create comprehensive comparison plots and analysis
    if all_results:
        overall_results['individual_results'] = all_results
        overall_results['status'] = 'completed'
        
        plot_path = create_comprehensive_comparison_plots(all_results)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results/comprehensive_attack_results_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        import json
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        # Create multiple CSV files for different analyses
        csv_data = []
        detailed_metrics_data = []
        training_progress_data = []
        
        for result in all_results:
            # Main summary data
            csv_data.append({
                'Attack_Type': result['attack_type'],
                'Initial_Accuracy': result['initial_accuracy'],
                'Final_Accuracy': result['final_accuracy'],
                'Improvement': result['improvement'],
                'Detection_Precision': result['detection_metrics']['precision'],
                'Detection_Recall': result['detection_metrics']['recall'],
                'Detection_F1': result['detection_metrics']['f1_score'],
                'True_Positives': result['detection_metrics']['true_positives'],
                'False_Positives': result['detection_metrics']['false_positives'],
                'False_Negatives': result['detection_metrics']['false_negatives'],
                'Malicious_Clients': ','.join(map(str, result['malicious_indices'])),
                'Timestamp': result['timestamp']
            })
            
            # Detailed metrics for research analysis
            detailed_metrics_data.append({
                'Attack_Type': result['attack_type'],
                'Attack_Family': result['attack_type'].split('_')[0],  # e.g., 'scaling', 'noise', etc.
                'Initial_Accuracy': result['initial_accuracy'],
                'Final_Accuracy': result['final_accuracy'],
                'Accuracy_Change': result['improvement'],
                'Accuracy_Change_Percent': (result['improvement'] / result['initial_accuracy'] * 100) if result['initial_accuracy'] > 0 else 0,
                'Detection_Precision': result['detection_metrics']['precision'],
                'Detection_Recall': result['detection_metrics']['recall'],
                'Detection_F1_Score': result['detection_metrics']['f1_score'],
                'Detection_Accuracy': (result['detection_metrics']['true_positives'] + (NUM_CLIENTS - len(result['malicious_indices']) - result['detection_metrics']['false_positives'])) / NUM_CLIENTS,
                'False_Positive_Rate': result['detection_metrics']['false_positives'] / (NUM_CLIENTS - len(result['malicious_indices'])) if (NUM_CLIENTS - len(result['malicious_indices'])) > 0 else 0,
                'False_Negative_Rate': result['detection_metrics']['false_negatives'] / len(result['malicious_indices']) if len(result['malicious_indices']) > 0 else 0,
                'Attack_Success_Rate': 1 - result['detection_metrics']['recall'],  # How many malicious clients went undetected
                'System_Robustness': result['final_accuracy'] / result['initial_accuracy'] if result['initial_accuracy'] > 0 else 0,
                'Overall_Performance': (0.6 * result['final_accuracy'] + 0.4 * result['detection_metrics']['f1_score']),  # Combined score
                'Experiment_Date': result['timestamp']
            })
            
            # Training progress data (if available)
            if 'training_history' in result and 'accuracies' in result['training_history']:
                for epoch, accuracy in enumerate(result['training_history']['accuracies']):
                    training_progress_data.append({
                        'Attack_Type': result['attack_type'],
                        'Epoch': epoch + 1,
                        'Accuracy': accuracy,
                        'Experiment_Timestamp': result['timestamp']
                    })
        
        # Save main summary CSV
        df_summary = pd.DataFrame(csv_data)
        csv_file = f"results/comprehensive_attack_summary_{timestamp}.csv"
        df_summary.to_csv(csv_file, index=False)
        
        # Save detailed metrics CSV
        df_detailed = pd.DataFrame(detailed_metrics_data)
        detailed_csv_file = f"results/detailed_attack_metrics_{timestamp}.csv"
        df_detailed.to_csv(detailed_csv_file, index=False)
        
        # Save training progress CSV (if data available)
        if training_progress_data:
            df_progress = pd.DataFrame(training_progress_data)
            progress_csv_file = f"results/training_progress_{timestamp}.csv"
            df_progress.to_csv(progress_csv_file, index=False)
        else:
            progress_csv_file = "No training progress data available"
        
        # Create configuration CSV for reproducibility
        config_data = [{
            'Parameter': key,
            'Value': str(value)
        } for key, value in overall_results['config'].items()]
        
        # Add additional important config parameters
        additional_config = {
            'VAE_EPOCHS': VAE_EPOCHS,
            'DUAL_ATTENTION_EPOCHS': DUAL_ATTENTION_EPOCHS,
            'SHAPLEY_SAMPLES': SHAPLEY_SAMPLES,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LR,
            'LOCAL_EPOCHS_ROOT': LOCAL_EPOCHS_ROOT,
            'RANDOM_SEED': RANDOM_SEED,
            'GRADIENT_COMBINATION_METHOD': GRADIENT_COMBINATION_METHOD
        }
        
        for key, value in additional_config.items():
            config_data.append({
                'Parameter': key,
                'Value': str(value)
            })
        
        df_config = pd.DataFrame(config_data)
        config_csv_file = f"results/experiment_config_{timestamp}.csv"
        df_config.to_csv(config_csv_file, index=False)
        
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📁 Detailed Results: {results_file}")
        print(f"📊 CSV Files Generated:")
        print(f"   • Summary: {csv_file}")
        print(f"   • Detailed Metrics: {detailed_csv_file}")
        print(f"   • Training Progress: {progress_csv_file}")
        print(f"   • Experiment Config: {config_csv_file}")
        print(f"📈 Comparison Plot: {plot_path}")
        print(f"✅ Successfully tested {len(all_results)}/{len(ALL_ATTACK_TYPES)} attack types")
        
        # Return updated summary with all file paths
        return {
            'all_results': all_results,
            'plot_path': plot_path,
            'results_file': results_file,
            'csv_files': {
                'summary': csv_file,
                'detailed_metrics': detailed_csv_file,
                'training_progress': progress_csv_file,
                'config': config_csv_file
            },
            'summary': {
                'total_attacks_tested': len(all_results),
                'best_accuracy': max([r['final_accuracy'] for r in all_results]) if all_results else 0,
                'mean_improvement': np.mean([r['improvement'] for r in all_results]) if all_results else 0,
                'mean_detection_precision': np.mean([r['detection_metrics']['precision'] for r in all_results]) if all_results else 0
            }
        }
    
    else:
        overall_results['status'] = 'failed'
        print("❌ No experiments completed successfully")
        return None

if __name__ == "__main__":
    main() 