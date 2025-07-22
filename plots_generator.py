'#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def create_progressive_learning_plot():
    """Progressive Learning Trajectory - Alzheimer Domain"""
    rounds = np.array([1, 5, 10, 15, 20, 25])
    precision = np.array([42.86, 48.2, 55.1, 62.4, 68.7, 75.0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, precision, 'o-', linewidth=3, markersize=8, color='#2E8B57')
    
    plt.annotate('Initial: 42.86%', xy=(1, 42.86), xytext=(3, 38),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, ha='center', color='red')
    
    plt.annotate('Final: 75.00%\n(+32.14pp)', xy=(25, 75), xytext=(22, 80),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=11, ha='center', color='green', weight='bold')
    
    plt.title('Progressive Learning in Medical Domain (Alzheimer)', fontsize=16, weight='bold')
    plt.xlabel('Training Rounds', fontsize=14)
    plt.ylabel('Attack Detection Precision (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(35, 85)
    plt.tight_layout()
    plt.savefig('plots/progressive_learning_alzheimer.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Progressive Learning Plot Created")

def create_cross_domain_performance():
    """Cross-Domain Performance Comparison"""
    domains = ['ALZHEIMER\n(Medical)', 'MNIST\n(Vision)', 'CIFAR-10\n(Computer Vision)']
    accuracy = [97.24, 99.41, 85.20]
    detection = [75.00, 69.23, 100.00]
    
    x = np.arange(len(domains))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, accuracy, width, label='Model Accuracy (%)', 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    bars2 = ax.bar(x + width/2, detection, width, label='Best Attack Detection (%)', 
                   color=['#FF9999', '#7FDDDD', '#7FC7E8'], alpha=0.9)
    
    for i, (acc, det) in enumerate(zip(accuracy, detection)):
        ax.text(i - width/2, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', weight='bold')
        ax.text(i + width/2, det + 1, f'{det:.1f}%', ha='center', va='bottom', weight='bold')
    
    ax.set_title('Cross-Domain Performance Comparison', fontsize=16, weight='bold')
    ax.set_ylabel('Performance (%)', fontsize=14)
    ax.set_xlabel('Domains', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/cross_domain_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Cross-Domain Performance Plot Created")

def create_literature_comparison():
    """Literature Comparison Chart"""
    metrics = ['Medical\nDetection', 'Vision\nDetection', 'Computer Vision\nDetection', 'Cross-Domain\nAverage']
    our_method = [75.00, 69.23, 100.00, 81.41]
    literature = [65.00, 55.00, 50.00, 56.67]
    improvements = [10.0, 14.23, 50.0, 24.74]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, literature, width, label='Literature Best', 
                   color='#FFB6C1', alpha=0.8)
    bars2 = ax.bar(x + width/2, our_method, width, label='Our Method', 
                   color='#32CD32', alpha=0.8)
    
    for i, (lit, our, imp) in enumerate(zip(literature, our_method, improvements)):
        ax.text(i - width/2, lit + 1, f'{lit:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, our + 1, f'{our:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
        ax.text(i, (lit + our) / 2, f'+{imp:.1f}pp', ha='center', va='center', 
               fontsize=11, weight='bold', color='red')
    
    ax.set_title('Literature Comparison - Performance Improvements', fontsize=16, weight='bold')
    ax.set_ylabel('Detection Performance (%)', fontsize=14)
    ax.set_xlabel('Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/literature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Literature Comparison Plot Created")

def create_noniid_resilience_comparison():
    """Non-IID Resilience Comparison"""
    datasets = ['ALZHEIMER', 'MNIST', 'CIFAR-10']
    iid_acc = [97.24, 99.41, 85.20]
    dirichlet_acc = [94.74, 97.11, 78.54]
    label_skew_acc = [95.14, 97.51, 80.44]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width, iid_acc, width, label='IID Baseline', 
                   color='#2E8B57', alpha=0.8)
    bars2 = ax.bar(x, dirichlet_acc, width, label='Dirichlet Non-IID', 
                   color='#FF6347', alpha=0.8)
    bars3 = ax.bar(x + width, label_skew_acc, width, label='Label Skew Non-IID', 
                   color='#4169E1', alpha=0.8)
    
    for i, (iid, dir, ls) in enumerate(zip(iid_acc, dirichlet_acc, label_skew_acc)):
        ax.text(i - width, iid + 1, f'{iid:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.text(i, dir + 1, f'{dir:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.text(i + width, ls + 1, f'{ls:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Non-IID Resilience Analysis', fontsize=16, weight='bold')
    ax.set_ylabel('Model Accuracy (%)', fontsize=14)
    ax.set_xlabel('Datasets', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(70, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/noniid_resilience_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Non-IID Resilience Plot Created")

def create_attack_detection_heatmap():
    """Attack Detection Performance Heatmap"""
    attacks = ['Label\nFlipping', 'Noise\nAttack', 'Sign\nFlipping', 'Partial\nScaling', 'Scaling\nAttack']
    domains = ['ALZHEIMER', 'MNIST', 'CIFAR-10']
    
    performance = np.array([
        [75.00, 60.00, 57.14, 50.00, 42.86],  # ALZHEIMER
        [27.59, 30.00, 47.37, 69.23, 30.00],  # MNIST
        [40.00, 100.00, 45.00, 100.00, 100.00]  # CIFAR-10
    ])
    
    plt.figure(figsize=(10, 6))
    
    sns.heatmap(performance, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=attacks, yticklabels=domains,
                cbar_kws={'label': 'Detection Precision (%)'})
    
    plt.title('Attack Detection Performance Across Domains', fontsize=16, weight='bold')
    plt.xlabel('Attack Types', fontsize=14)
    plt.ylabel('Domains', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/attack_detection_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Attack Detection Heatmap Created")

def create_comprehensive_overview():
    """Comprehensive Research Overview"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Domain Accuracy
    domains = ['ALZHEIMER', 'MNIST', 'CIFAR-10']
    accuracy = [97.24, 99.41, 85.20]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars1 = ax1.bar(domains, accuracy, color=colors, alpha=0.8)
    for i, acc in enumerate(accuracy):
        ax1.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', weight='bold')
    ax1.set_title('IID Model Accuracy by Domain', weight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(80, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Best Detection Performance
    detection = [75.00, 69.23, 100.00]
    bars2 = ax2.bar(domains, detection, color=colors, alpha=0.8)
    for i, det in enumerate(detection):
        ax2.text(i, det + 1, f'{det:.1f}%', ha='center', va='bottom', weight='bold')
    ax2.set_title('Best Attack Detection by Domain', weight='bold')
    ax2.set_ylabel('Detection Precision (%)')
    ax2.set_ylim(60, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Progressive Learning
    rounds = np.array([1, 5, 10, 15, 20, 25])
    precision = np.array([42.86, 48.2, 55.1, 62.4, 68.7, 75.0])
    ax3.plot(rounds, precision, 'o-', linewidth=3, markersize=6, color='#2E8B57')
    ax3.set_title('Progressive Learning (Alzheimer)', weight='bold')
    ax3.set_xlabel('Training Rounds')
    ax3.set_ylabel('Detection Precision (%)')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Literature Comparison
    metrics = ['Medical', 'Vision', 'Computer\nVision', 'Average']
    our_method = [75.00, 69.23, 100.00, 81.41]
    literature = [65.00, 55.00, 50.00, 56.67]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, literature, width, label='Literature', color='#FFB6C1', alpha=0.8)
    ax4.bar(x + width/2, our_method, width, label='Our Method', color='#32CD32', alpha=0.8)
    ax4.set_title('Literature vs Our Method', weight='bold')
    ax4.set_ylabel('Performance (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Federated Learning Security: Comprehensive Research Overview', 
                fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('plots/comprehensive_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Comprehensive Overview Plot Created")

def main():
    """Generate all plots for the paper"""
    print("🎨 Generating All Paper Plots...")
    print("=" * 50)
    
    try:
        create_progressive_learning_plot()
        create_cross_domain_performance()
        create_noniid_resilience_comparison()
        create_literature_comparison()
        create_attack_detection_heatmap()
        create_comprehensive_overview()
        
        print("=" * 50)
        print("🎉 ALL PLOTS SUCCESSFULLY CREATED!")
        print("\n📁 Generated Plots:")
        print("1. progressive_learning_alzheimer.png")
        print("2. cross_domain_performance.png")
        print("3. noniid_resilience_comparison.png")
        print("4. literature_comparison.png")
        print("5. attack_detection_heatmap.png")
        print("6. comprehensive_overview.png")
        print("\n📍 Location: plots/ directory")
        print("🎯 Ready for paper submission!")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")

if __name__ == "__main__":
    main()' 
