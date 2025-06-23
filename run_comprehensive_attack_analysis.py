#!/usr/bin/env python3
"""
Comprehensive Attack Analysis for Federated Learning

This script systematically tests all attack types with different aggregation methods
while maintaining efficiency by training shared models (VAE, etc.) only once.

Features:
- Fixed configuration: MNIST + CNN + IID
- All attack types tested systematically  
- Three aggregation methods: hybrid, dual_attention, rl_actor_critic
- Shared model training (VAE trained once)
- Comprehensive comparison plots
- Reduced epochs for faster execution

Usage:
    python run_comprehensive_attack_analysis.py
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveAttackAnalyzer:
    """Comprehensive attack analysis with systematic testing and comparison plots."""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_attack_analysis"
        self.plots_dir = "research_plots/attack_comparison"
        self.progress_file = os.path.join(self.results_dir, "analysis_progress.json")
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Fixed configuration for controlled experiments
        self.base_config = {
            'DATASET': 'MNIST',
            'MODEL': 'CNN',
            'DATA_DISTRIBUTION': 'iid',
            'ENABLE_NON_IID': False,
            'NUM_CLIENTS': 10,
            'FRACTION_MALICIOUS': 0.3,
            # Reduced epochs for faster execution
            'VAE_EPOCHS': 30,          # کم شده از 50
            'GLOBAL_EPOCHS': 15,       # کم شده از 30  
            'LOCAL_EPOCHS_CLIENT': 5,  # کم شده از 10
            'LOCAL_EPOCHS_ROOT': 30,   # کم شده از 100
        }
        
        # Attack types to test
        self.attack_types = [
            'scaling_attack',
            'partial_scaling_attack', 
            'sign_flipping_attack',
            'noise_attack',
            'min_max_attack',
            'min_sum_attack',
            'targeted_attack',
            'label_flipping'
        ]
        
        # Aggregation methods to test
        self.aggregation_methods = [
            'hybrid',           # شروع با dual attention -> RL
            'dual_attention',   # فقط dual attention
            'rl_actor_critic'   # فقط RL
        ]
        
        # Results storage
        self.all_results = []
        self.shared_models_trained = False
        
        print(f"🧪 COMPREHENSIVE ATTACK ANALYSIS INITIALIZED")
        print(f"📊 Total experiments: {len(self.attack_types)} attacks × {len(self.aggregation_methods)} methods = {len(self.attack_types) * len(self.aggregation_methods)}")
        print(f"⏱️  Estimated time with optimizations: ~3-4 hours")
    
    def run_comprehensive_analysis(self):
        """Run comprehensive attack analysis across all combinations."""
        
        print(f"\n🚀 STARTING COMPREHENSIVE ATTACK ANALYSIS")
        print(f"{'='*60}")
        
        start_time = time.time()
        experiment_count = 0
        total_experiments = len(self.attack_types) * len(self.aggregation_methods)
        
        # Run experiments for each aggregation method
        for aggregation_method in self.aggregation_methods:
            
            print(f"\n🔧 TESTING AGGREGATION METHOD: {aggregation_method.upper()}")
            print(f"{'='*50}")
            
            method_results = []
            
            # Run all attacks for this aggregation method
            for attack_type in self.attack_types:
                experiment_count += 1
                
                print(f"\n⚔️  EXPERIMENT {experiment_count}/{total_experiments}")
                print(f"🎯 Attack: {attack_type}")
                print(f"🔧 Aggregation: {aggregation_method}")
                
                try:
                    # Configure and run experiment
                    result = self._run_single_experiment(attack_type, aggregation_method)
                    
                    if result:
                        result['aggregation_method'] = aggregation_method
                        result['attack_type'] = attack_type
                        result['experiment_id'] = f"{aggregation_method}_{attack_type}"
                        
                        method_results.append(result)
                        self.all_results.append(result)
                        
                        print(f"✅ Success: Final Accuracy = {result.get('final_accuracy', 0):.4f}")
                        print(f"🎯 Detection: Precision = {result.get('detection_metrics', {}).get('precision', 0):.3f}")
                    
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    # Log error but continue
                    error_result = {
                        'aggregation_method': aggregation_method,
                        'attack_type': attack_type,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.all_results.append(error_result)
            
            # Create method-specific analysis
            if method_results:
                self._create_method_analysis(aggregation_method, method_results)
        
        # Create comprehensive comparison
        total_time = (time.time() - start_time) / 3600
        
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print(f"⏱️  Total time: {total_time:.2f} hours")
        print(f"✅ Successful experiments: {len([r for r in self.all_results if r.get('status') != 'failed'])}")
        print(f"❌ Failed experiments: {len([r for r in self.all_results if r.get('status') == 'failed'])}")
        
        # Generate final comprehensive comparison
        self._create_comprehensive_comparison()
        
        # Save all results
        self._save_comprehensive_results()
    
    def _run_single_experiment(self, attack_type: str, aggregation_method: str) -> Dict[str, Any]:
        """Run a single experiment with specified attack and aggregation method."""
        
        # Configure the experiment
        config_updates = self.base_config.copy()
        config_updates.update({
            'ATTACK_TYPE': attack_type,
            'RL_AGGREGATION_METHOD': aggregation_method,
            'ENABLE_ATTACK_SIMULATION': True
        })
        
        # Apply configuration
        self._apply_config_updates(config_updates)
        
        # Run the experiment
        experiment_start = time.time()
        
        try:
            # Clear main module to ensure fresh import
            if 'main' in sys.modules:
                del sys.modules['main']
            
            # Import and run main
            from main import main
            result = main()
            
            # Add execution info
            execution_time = (time.time() - experiment_start) / 60
            result['execution_time_minutes'] = execution_time
            result['status'] = 'completed'
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            print(f"❌ Experiment failed: {str(e)}")
            raise e
    
    def _apply_config_updates(self, config_updates: Dict[str, Any]):
        """Apply configuration updates."""
        
        # Write temporary config override
        override_file = 'temp_config_override.py'
        with open(override_file, 'w') as f:
            f.write("# Temporary configuration overrides\n")
            f.write("# Auto-generated for comprehensive attack analysis\n\n")
            
            for key, value in config_updates.items():
                if isinstance(value, str):
                    f.write(f"{key} = '{value}'\n")
                elif isinstance(value, bool):
                    f.write(f"{key} = {value}\n")
                else:
                    f.write(f"{key} = {value}\n")
        
        print(f"📝 Applied {len(config_updates)} configuration updates")
    
    def _create_method_analysis(self, method: str, results: List[Dict[str, Any]]):
        """Create analysis plots for specific aggregation method."""
        
        if len(results) < 2:
            print(f"⚠️  Insufficient results for {method} analysis")
            return
        
        # Extract data
        attack_names = [r['attack_type'] for r in results]
        final_accuracies = [r.get('final_accuracy', 0) for r in results]
        improvements = [r.get('final_accuracy', 0) - r.get('initial_accuracy', 0) for r in results]
        detection_precisions = [r.get('detection_metrics', {}).get('precision', 0) for r in results]
        detection_recalls = [r.get('detection_metrics', {}).get('recall', 0) for r in results]
        
        # Create method-specific plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Attack Analysis: {method.upper()}', fontsize=16, fontweight='bold')
        
        # Plot 1: Final Accuracy by Attack
        axes[0, 0].bar(range(len(results)), final_accuracies, alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Final Accuracy by Attack Type')
        axes[0, 0].set_ylabel('Final Accuracy')
        axes[0, 0].set_xticks(range(len(results)))
        axes[0, 0].set_xticklabels([name.replace('_attack', '') for name in attack_names], rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Accuracy Improvement
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[0, 1].bar(range(len(results)), improvements, alpha=0.7, color=colors)
        axes[0, 1].set_title('Accuracy Improvement by Attack')
        axes[0, 1].set_ylabel('Accuracy Improvement')
        axes[0, 1].set_xticks(range(len(results)))
        axes[0, 1].set_xticklabels([name.replace('_attack', '') for name in attack_names], rotation=45, ha='right')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Detection Performance
        x_pos = np.arange(len(results))
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, detection_precisions, width, alpha=0.7, color='orange', label='Precision')
        axes[1, 0].bar(x_pos + width/2, detection_recalls, width, alpha=0.7, color='purple', label='Recall')
        axes[1, 0].set_title('Detection Performance by Attack')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([name.replace('_attack', '') for name in attack_names], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Performance vs Detection Trade-off
        axes[1, 1].scatter(improvements, detection_precisions, alpha=0.7, s=100, c='red')
        for i, attack in enumerate(attack_names):
            axes[1, 1].annotate(attack.replace('_attack', ''), 
                              (improvements[i], detection_precisions[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_title('Accuracy vs Detection Trade-off')
        axes[1, 1].set_xlabel('Accuracy Improvement')
        axes[1, 1].set_ylabel('Detection Precision')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save method-specific plot
        plot_path = os.path.join(self.plots_dir, f"{method}_attack_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 {method} analysis plot saved: {plot_path}")
    
    def _create_comprehensive_comparison(self):
        """Create comprehensive comparison plots across all methods and attacks."""
        
        successful_results = [r for r in self.all_results if r.get('status') != 'failed']
        
        if len(successful_results) < 6:
            print("⚠️  Insufficient results for comprehensive comparison")
            return
        
        # Prepare data for comparison
        df_data = []
        for result in successful_results:
            row = {
                'aggregation': result.get('aggregation_method', 'unknown'),
                'attack': result.get('attack_type', 'unknown').replace('_attack', ''),
                'final_accuracy': result.get('final_accuracy', 0),
                'initial_accuracy': result.get('initial_accuracy', 0),
                'improvement': result.get('final_accuracy', 0) - result.get('initial_accuracy', 0),
                'detection_precision': result.get('detection_metrics', {}).get('precision', 0),
                'detection_recall': result.get('detection_metrics', {}).get('recall', 0),
                'detection_f1': result.get('detection_metrics', {}).get('f1_score', 0)
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Create comprehensive comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('COMPREHENSIVE ATTACK & AGGREGATION COMPARISON', fontsize=18, fontweight='bold')
        
        # Plot 1: Accuracy Improvement by Method and Attack
        pivot_improvement = df.pivot(index='attack', columns='aggregation', values='improvement')
        im1 = axes[0, 0].imshow(pivot_improvement.values, cmap='RdYlGn', aspect='auto')
        axes[0, 0].set_title('Accuracy Improvement Heatmap')
        axes[0, 0].set_xticks(range(len(pivot_improvement.columns)))
        axes[0, 0].set_xticklabels(pivot_improvement.columns, rotation=45)
        axes[0, 0].set_yticks(range(len(pivot_improvement.index)))
        axes[0, 0].set_yticklabels(pivot_improvement.index)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Detection Precision by Method and Attack
        pivot_precision = df.pivot(index='attack', columns='aggregation', values='detection_precision')
        im2 = axes[0, 1].imshow(pivot_precision.values, cmap='Blues', aspect='auto')
        axes[0, 1].set_title('Detection Precision Heatmap')
        axes[0, 1].set_xticks(range(len(pivot_precision.columns)))
        axes[0, 1].set_xticklabels(pivot_precision.columns, rotation=45)
        axes[0, 1].set_yticks(range(len(pivot_precision.index)))
        axes[0, 1].set_yticklabels(pivot_precision.index)
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot 3: Method Performance Summary
        method_summary = df.groupby('aggregation').agg({
            'improvement': 'mean',
            'detection_precision': 'mean',
            'detection_f1': 'mean'
        })
        
        x_pos = np.arange(len(method_summary))
        width = 0.25
        axes[0, 2].bar(x_pos - width, method_summary['improvement'], width, label='Avg Improvement', alpha=0.7)
        axes[0, 2].bar(x_pos, method_summary['detection_precision'], width, label='Avg Precision', alpha=0.7)
        axes[0, 2].bar(x_pos + width, method_summary['detection_f1'], width, label='Avg F1-Score', alpha=0.7)
        axes[0, 2].set_title('Method Performance Summary')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(method_summary.index, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Attack Difficulty Analysis
        attack_summary = df.groupby('attack').agg({
            'improvement': ['mean', 'std'],
            'detection_precision': ['mean', 'std']
        })
        attack_summary.columns = ['_'.join(col).strip() for col in attack_summary.columns]
        
        attacks = attack_summary.index
        x_pos = np.arange(len(attacks))
        axes[1, 0].errorbar(x_pos, attack_summary['improvement_mean'], 
                          yerr=attack_summary['improvement_std'], 
                          fmt='o', capsize=5, capthick=2)
        axes[1, 0].set_title('Attack Impact on Model Performance')
        axes[1, 0].set_xlabel('Attack Type')
        axes[1, 0].set_ylabel('Mean Accuracy Improvement')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(attacks, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Detection Effectiveness
        axes[1, 1].errorbar(x_pos, attack_summary['detection_precision_mean'],
                          yerr=attack_summary['detection_precision_std'],
                          fmt='s', capsize=5, capthick=2, color='orange')
        axes[1, 1].set_title('Detection Effectiveness by Attack')
        axes[1, 1].set_xlabel('Attack Type')
        axes[1, 1].set_ylabel('Mean Detection Precision')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(attacks, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Overall Performance Ranking
        # Calculate combined score (weighted accuracy + detection)
        df['combined_score'] = 0.6 * df['improvement'] + 0.4 * df['detection_precision']
        ranking = df.groupby(['aggregation', 'attack'])['combined_score'].mean().reset_index()
        ranking = ranking.sort_values('combined_score', ascending=False)
        
        axes[1, 2].barh(range(len(ranking.head(10))), ranking.head(10)['combined_score'], alpha=0.7)
        axes[1, 2].set_title('Top 10 Method-Attack Combinations')
        axes[1, 2].set_xlabel('Combined Performance Score')
        axes[1, 2].set_yticks(range(len(ranking.head(10))))
        axes[1, 2].set_yticklabels([f"{row['aggregation']}\n{row['attack']}" for _, row in ranking.head(10).iterrows()])
        axes[1, 2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save comprehensive comparison
        plot_path = os.path.join(self.plots_dir, f"comprehensive_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Comprehensive comparison plot saved: {plot_path}")
        
        # Print summary statistics
        self._print_summary_statistics(df)
    
    def _print_summary_statistics(self, df: pd.DataFrame):
        """Print comprehensive summary statistics."""
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Overall statistics
        print(f"🔢 Total successful experiments: {len(df)}")
        print(f"📈 Mean accuracy improvement: {df['improvement'].mean():.4f} ± {df['improvement'].std():.4f}")
        print(f"🎯 Mean detection precision: {df['detection_precision'].mean():.3f} ± {df['detection_precision'].std():.3f}")
        print(f"🏆 Best improvement: {df['improvement'].max():.4f}")
        print(f"🎯 Best detection: {df['detection_precision'].max():.3f}")
        
        # Method comparison
        print(f"\n🔧 BY AGGREGATION METHOD:")
        method_stats = df.groupby('aggregation').agg({
            'improvement': ['mean', 'std', 'count'],
            'detection_precision': ['mean', 'std']
        }).round(4)
        print(method_stats)
        
        # Attack analysis
        print(f"\n⚔️  BY ATTACK TYPE:")
        attack_stats = df.groupby('attack').agg({
            'improvement': ['mean', 'std'],
            'detection_precision': ['mean', 'std']
        }).round(4)
        print(attack_stats)
        
        # Best combinations
        df['combined_score'] = 0.6 * df['improvement'] + 0.4 * df['detection_precision']
        best_combinations = df.nlargest(5, 'combined_score')[['aggregation', 'attack', 'improvement', 'detection_precision', 'combined_score']]
        print(f"\n🏆 TOP 5 COMBINATIONS:")
        print(best_combinations.round(4))
    
    def _save_comprehensive_results(self):
        """Save all results to files."""
        
        # Save raw results
        results_file = os.path.join(self.results_dir, f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        # Save DataFrame
        successful_results = [r for r in self.all_results if r.get('status') != 'failed']
        if successful_results:
            df_data = []
            for result in successful_results:
                row = {
                    'experiment_id': result.get('experiment_id', 'unknown'),
                    'aggregation_method': result.get('aggregation_method', 'unknown'),
                    'attack_type': result.get('attack_type', 'unknown'),
                    'final_accuracy': result.get('final_accuracy', 0),
                    'initial_accuracy': result.get('initial_accuracy', 0),
                    'improvement': result.get('final_accuracy', 0) - result.get('initial_accuracy', 0),
                    'detection_precision': result.get('detection_metrics', {}).get('precision', 0),
                    'detection_recall': result.get('detection_metrics', {}).get('recall', 0),
                    'detection_f1': result.get('detection_metrics', {}).get('f1_score', 0),
                    'execution_time_minutes': result.get('execution_time_minutes', 0),
                    'timestamp': result.get('timestamp', '')
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_file = os.path.join(self.results_dir, f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(csv_file, index=False)
            
            print(f"📁 Results saved:")
            print(f"   JSON: {results_file}")
            print(f"   CSV:  {csv_file}")

def main():
    """Main function for comprehensive attack analysis."""
    
    print(f"🧪 COMPREHENSIVE ATTACK ANALYSIS")
    print(f"{'='*60}")
    print(f"📋 Configuration:")
    print(f"   Dataset: MNIST")
    print(f"   Model: CNN") 
    print(f"   Data Distribution: IID")
    print(f"   Attack Types: 8 types")
    print(f"   Aggregation Methods: 3 methods")
    print(f"   Total Experiments: 24")
    print(f"   Reduced Epochs for Speed")
    
    # Confirm execution
    confirm = input(f"\n⚠️  This will run 24 experiments (~3-4 hours). Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Analysis cancelled")
        return
    
    # Initialize and run analyzer
    analyzer = ComprehensiveAttackAnalyzer()
    analyzer.run_comprehensive_analysis()
    
    print(f"\n🎉 COMPREHENSIVE ATTACK ANALYSIS COMPLETED!")
    print(f"📊 Check results in: results/comprehensive_attack_analysis/")
    print(f"📈 Check plots in: research_plots/attack_comparison/")

if __name__ == "__main__":
    main() 