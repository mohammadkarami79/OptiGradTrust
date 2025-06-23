"""
Comprehensive plotting utilities for federated learning research.
Creates publication-quality plots with descriptive naming for paper use.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Any, Optional

# Set publication-quality defaults
plt.style.use('default')
sns.set_palette("husl")

# Create plots directory
PLOTS_DIR = "research_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def create_experiment_timestamp():
    """Create a timestamp for experiment identification."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_plot(fig, name: str, experiment_id: str = None, dpi: int = 300):
    """Save plot with descriptive naming for research."""
    if experiment_id is None:
        experiment_id = create_experiment_timestamp()
    
    filename = f"{PLOTS_DIR}/{experiment_id}_{name}.png"
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"📊 Plot saved: {filename}")
    return filename

def plot_training_progress(round_metrics: Dict, experiment_config: Dict, 
                         experiment_id: str = None) -> str:
    """Create comprehensive training progress visualization."""
    
    if experiment_id is None:
        experiment_id = create_experiment_timestamp()
    
    # Extract data
    rounds = sorted([k for k in round_metrics.keys() if isinstance(k, (int, str)) and str(k).isdigit()])
    accuracies = [round_metrics[r].get('accuracy', 0) for r in rounds]
    losses = [round_metrics[r].get('loss', 0) for r in rounds]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy over rounds
    ax1.plot(rounds, accuracies, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(accuracies) - 0.001, max(accuracies) + 0.001])
    
    # Plot 2: Loss over rounds
    ax2.plot(rounds, losses, 'r-', linewidth=2, marker='s', markersize=6)
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Model Loss Progression', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Improvement metrics
    improvements = [acc - accuracies[0] for acc in accuracies]
    ax3.bar(rounds, improvements, alpha=0.7, color='green')
    ax3.set_xlabel('Communication Round', fontsize=12)
    ax3.set_ylabel('Accuracy Improvement', fontsize=12)
    ax3.set_title('Cumulative Accuracy Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Configuration summary
    ax4.axis('off')
    config_text = f"""Experiment Configuration:
Dataset: {experiment_config.get('DATASET', 'N/A')}
Model: {experiment_config.get('MODEL', 'N/A')}
Clients: {experiment_config.get('NUM_CLIENTS', 'N/A')}
Malicious: {experiment_config.get('FRACTION_MALICIOUS', 'N/A')}
Attack: {experiment_config.get('ATTACK_TYPE', 'N/A')}
Method: {experiment_config.get('AGGREGATION_METHOD', 'N/A')}

Final Results:
Initial Accuracy: {accuracies[0]:.4f}
Final Accuracy: {accuracies[-1]:.4f}
Total Improvement: {improvements[-1]:.4f}
Total Rounds: {len(rounds)}"""
    
    ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    return save_plot(fig, "training_progress_comprehensive", experiment_id)

def plot_trust_scores(round_metrics: Dict, clients_info: List, 
                     experiment_id: str = None) -> str:
    """Create trust score analysis visualization."""
    
    if experiment_id is None:
        experiment_id = create_experiment_timestamp()
    
    # Get final round data
    final_round = max([k for k in round_metrics.keys() if isinstance(k, (int, str))])
    final_metrics = round_metrics[final_round]
    trust_scores = final_metrics.get('trust_scores', {})
    
    # Organize data
    malicious_indices = [i for i, client in enumerate(clients_info) if client.is_malicious]
    honest_indices = [i for i, client in enumerate(clients_info) if not client.is_malicious]
    
    malicious_scores = [trust_scores.get(i, 0.5) for i in malicious_indices]
    honest_scores = [trust_scores.get(i, 0.5) for i in honest_indices]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trust scores by client
    all_clients = list(range(len(clients_info)))
    all_scores = [trust_scores.get(i, 0.5) for i in all_clients]
    colors = ['red' if clients_info[i].is_malicious else 'blue' for i in all_clients]
    
    bars = ax1.bar(all_clients, all_scores, color=colors, alpha=0.7)
    ax1.set_xlabel('Client ID', fontsize=12)
    ax1.set_ylabel('Trust Score', fontsize=12)
    ax1.set_title('Trust Scores by Client', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Malicious', 'Honest'], loc='upper right')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, all_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Distribution comparison
    ax2.hist(honest_scores, bins=10, alpha=0.7, label='Honest Clients', color='blue', density=True)
    ax2.hist(malicious_scores, bins=10, alpha=0.7, label='Malicious Clients', color='red', density=True)
    ax2.set_xlabel('Trust Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Trust Score Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"""Statistics:
Honest: μ={np.mean(honest_scores):.3f}, σ={np.std(honest_scores):.3f}
Malicious: μ={np.mean(malicious_scores):.3f}, σ={np.std(malicious_scores):.3f}
Separation: {np.mean(honest_scores) - np.mean(malicious_scores):.3f}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.tight_layout()
    return save_plot(fig, "trust_scores_analysis", experiment_id)

def plot_detection_metrics(detection_metrics: Dict, experiment_id: str = None) -> str:
    """Create detection performance visualization."""
    
    if experiment_id is None:
        experiment_id = create_experiment_timestamp()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Main metrics bar chart
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [detection_metrics.get('precision', 0),
              detection_metrics.get('recall', 0), 
              detection_metrics.get('f1_score', 0),
              detection_metrics.get('accuracy', 0)]
    
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Detection Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Confusion matrix
    tp = detection_metrics.get('true_positives', 0)
    fp = detection_metrics.get('false_positives', 0)
    fn = detection_metrics.get('false_negatives', 0)
    tn = detection_metrics.get('true_negatives', 0)
    
    confusion_matrix = np.array([[tp, fp], [fn, tn]])
    im = ax2.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Malicious', 'Honest'])
    ax2.set_yticklabels(['Malicious', 'Honest'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, confusion_matrix[i, j], ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black",
                    fontsize=16, fontweight='bold')
    
    # Plot 3: Trust score comparison
    honest_trust = detection_metrics.get('avg_trust_honest', 0.5)
    malicious_trust = detection_metrics.get('avg_trust_malicious', 0.5)
    
    categories = ['Honest Clients', 'Malicious Clients']
    trust_values = [honest_trust, malicious_trust]
    colors = ['blue', 'red']
    
    bars = ax3.bar(categories, trust_values, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Trust Score', fontsize=12)
    ax3.set_title('Average Trust Scores by Client Type', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    for bar, value in zip(bars, trust_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: Performance summary
    ax4.axis('off')
    
    # Calculate additional metrics
    false_positive_rate = detection_metrics.get('false_positive_rate', 0)
    detection_rate = detection_metrics.get('malicious_detection_rate', 0)
    
    summary_text = f"""Detection Performance Summary:

✅ True Positives: {tp}
❌ False Positives: {fp}  
❌ False Negatives: {fn}
✅ True Negatives: {tn}

📊 Key Metrics:
• Precision: {values[0]:.1%}
• Recall: {values[1]:.1%}  
• F1-Score: {values[2]:.1%}
• False Positive Rate: {false_positive_rate:.1%}

🎯 Trust Score Separation:
• Honest Avg: {honest_trust:.3f}
• Malicious Avg: {malicious_trust:.3f}
• Difference: {honest_trust - malicious_trust:.3f}

{'🎉 EXCELLENT DETECTION!' if values[0] > 0.9 and values[1] > 0.9 else '⚠️  NEEDS IMPROVEMENT' if values[0] < 0.7 or values[1] < 0.7 else '✅ GOOD DETECTION'}"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    return save_plot(fig, "detection_performance_comprehensive", experiment_id)

def plot_gradient_analysis(round_metrics: Dict, clients_info: List, 
                          experiment_id: str = None) -> str:
    """Create gradient analysis visualization."""
    
    if experiment_id is None:
        experiment_id = create_experiment_timestamp()
    
    # Get final round gradient data
    final_round = max([k for k in round_metrics.keys() if isinstance(k, (int, str))])
    final_metrics = round_metrics[final_round]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Gradient norms by client
    client_ids = list(range(len(clients_info)))
    gradient_norms = [final_metrics.get('gradient_norms', {}).get(i, 0) for i in client_ids]
    colors = ['red' if clients_info[i].is_malicious else 'blue' for i in client_ids]
    
    bars = ax1.bar(client_ids, gradient_norms, color=colors, alpha=0.7)
    ax1.set_xlabel('Client ID', fontsize=12)
    ax1.set_ylabel('Gradient Norm', fontsize=12)
    ax1.set_title('Gradient Norms by Client', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Malicious', 'Honest'], loc='upper right')
    
    # Plot 2: Weights distribution
    weights = [final_metrics.get('weights', {}).get(i, 0.1) for i in client_ids]
    
    bars = ax2.bar(client_ids, weights, color=colors, alpha=0.7)
    ax2.set_xlabel('Client ID', fontsize=12)
    ax2.set_ylabel('Aggregation Weight', fontsize=12)
    ax2.set_title('Final Aggregation Weights', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Attack impact analysis (if available)
    if any(clients_info[i].is_malicious for i in client_ids):
        malicious_norms = [gradient_norms[i] for i in client_ids if clients_info[i].is_malicious]
        honest_norms = [gradient_norms[i] for i in client_ids if not clients_info[i].is_malicious]
        
        ax3.hist(honest_norms, bins=10, alpha=0.7, label='Honest', color='blue', density=True)
        ax3.hist(malicious_norms, bins=10, alpha=0.7, label='Malicious', color='red', density=True)
        ax3.set_xlabel('Gradient Norm', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Gradient Norm Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Round-by-round accuracy
    rounds = sorted([k for k in round_metrics.keys() if isinstance(k, (int, str)) and str(k).isdigit()])
    accuracies = [round_metrics[r].get('accuracy', 0) for r in rounds]
    
    ax4.plot(rounds, accuracies, 'g-', linewidth=3, marker='o', markersize=8)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Learning Progression', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_plot(fig, "gradient_analysis_comprehensive", experiment_id)

def create_all_research_plots(results: Dict, clients_info: List = None, 
                            experiment_id: str = None) -> List[str]:
    """Create all publication-quality plots for research."""
    
    if experiment_id is None:
        experiment_id = create_experiment_timestamp()
    
    plot_files = []
    
    try:
        # 1. Training progress
        if 'round_metrics' in results and 'config' in results:
            plot_files.append(plot_training_progress(
                results['round_metrics'], results['config'], experiment_id))
        
        # 2. Trust scores (if clients_info available)
        if clients_info and 'round_metrics' in results:
            plot_files.append(plot_trust_scores(
                results['round_metrics'], clients_info, experiment_id))
        
        # 3. Detection metrics
        if 'detection_metrics' in results:
            plot_files.append(plot_detection_metrics(
                results['detection_metrics'], experiment_id))
        
        # 4. Gradient analysis (if clients_info available)
        if clients_info and 'round_metrics' in results:
            plot_files.append(plot_gradient_analysis(
                results['round_metrics'], clients_info, experiment_id))
        
        print(f"\n🎨 Created {len(plot_files)} publication-quality plots!")
        print(f"📁 All plots saved in: {PLOTS_DIR}/")
        print(f"🏷️  Experiment ID: {experiment_id}")
        
    except Exception as e:
        print(f"⚠️  Error creating plots: {str(e)}")
    
    return plot_files 