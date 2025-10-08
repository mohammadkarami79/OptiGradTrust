"""
COMPREHENSIVE EXPERIMENT SUITE FOR REVIEWER FEEDBACK
=====================================================

This master script runs ALL experiments required to address reviewer feedback:

CRITICAL EXPERIMENTS (Priority 1):
1. Ablation Study - Drop-one-feature analysis
2. Computational Overhead - Runtime & memory profiling
3. Fair Comparison - Baselines with FedBN-P
4. Extended Metrics - Precision, Recall, F1, AUC, Confusion Matrix
5. Confidence Intervals - Multiple runs with different seeds

IMPORTANT EXPERIMENTS (Priority 2):
6. Combined Attacks - Multiple simultaneous attacks
7. Scalability Analysis - Varying number of clients
8. Adversarial Ratios - Different percentages of malicious clients
9. Feature Correlation - Independence analysis
10. Extreme Heterogeneity - α=0.05, 0.01

ADDITIONAL ANALYSES (Priority 3):
11. Preprocessing Documentation
12. Statistical Significance Tests
13. Comprehensive Visualizations

Usage:
    python experiments/run_all_experiments.py --all
    python experiments/run_all_experiments.py --priority 1
    python experiments/run_all_experiments.py --quick  # Subset for testing
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import experiment modules (will create these)
from experiments import (
    ablation_study,
    combined_attacks,
    computational_overhead,
    confidence_intervals,
    extended_metrics,
    fair_comparison,
    scalability_tests,
    feature_correlation,
    extreme_heterogeneity,
    statistical_tests,
    visualization_suite,
    preprocessing_docs
)


class ExperimentRunner:
    """Master experiment runner for all reviewer feedback experiments."""
    
    def __init__(self, output_dir="experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_dir / f"session_{self.timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'session_id': self.timestamp,
            'start_time': datetime.now().isoformat(),
            'experiments': {}
        }
        
        print(f"🚀 Experiment Session: {self.timestamp}")
        print(f"📁 Output Directory: {self.session_dir}")
        print("="*80)
    
    def run_experiment(self, name, func, **kwargs):
        """Run a single experiment with error handling and timing."""
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT: {name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            result = func(output_dir=self.session_dir, **kwargs)
            duration = time.time() - start_time
            
            self.results['experiments'][name] = {
                'status': 'success',
                'duration_seconds': duration,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ {name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.results['experiments'][name] = {
                'status': 'failed',
                'duration_seconds': duration,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"❌ {name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_priority_1(self):
        """Run critical experiments (must-have for paper)."""
        print("\n" + "🎯"*40)
        print("PRIORITY 1: CRITICAL EXPERIMENTS")
        print("🎯"*40 + "\n")
        
        # 1. Ablation Study
        self.run_experiment(
            "Ablation Study",
            ablation_study.run_ablation_analysis,
            features_to_test=['vae', 'cosine_ref', 'cosine_peer', 'l2_norm', 'sign_consistency', 'shapley']
        )
        
        # 2. Computational Overhead
        self.run_experiment(
            "Computational Overhead",
            computational_overhead.profile_overhead,
            profile_memory=True,
            profile_time=True
        )
        
        # 3. Fair Comparison
        self.run_experiment(
            "Fair Comparison with FedBN-P",
            fair_comparison.run_fair_comparison,
            baselines=['FLGuard', 'FLTrust', 'FLAME']
        )
        
        # 4. Extended Metrics
        self.run_experiment(
            "Extended Metrics",
            extended_metrics.compute_extended_metrics,
            metrics=['precision', 'recall', 'f1', 'auc', 'confusion_matrix']
        )
        
        # 5. Confidence Intervals
        self.run_experiment(
            "Confidence Intervals",
            confidence_intervals.run_multiple_seeds,
            num_runs=5,
            seeds=[42, 123, 456, 789, 1024]
        )
    
    def run_priority_2(self):
        """Run important experiments (strengthen paper)."""
        print("\n" + "⭐"*40)
        print("PRIORITY 2: IMPORTANT EXPERIMENTS")
        print("⭐"*40 + "\n")
        
        # 6. Combined Attacks
        self.run_experiment(
            "Combined Attacks",
            combined_attacks.test_combined_attacks,
            combinations=[
                ['scaling_attack', 'noise_attack'],
                ['sign_flipping_attack', 'label_flipping'],
                ['partial_scaling_attack', 'noise_attack'],
                ['scaling_attack', 'sign_flipping_attack'],
                ['all_combined']  # worst case
            ]
        )
        
        # 7. Scalability - Clients
        self.run_experiment(
            "Scalability - Number of Clients",
            scalability_tests.test_varying_clients,
            client_counts=[10, 20, 50, 100]
        )
        
        # 8. Scalability - Adversarial Ratios
        self.run_experiment(
            "Scalability - Adversarial Ratios",
            scalability_tests.test_varying_adversarial_ratios,
            ratios=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        # 9. Feature Correlation
        self.run_experiment(
            "Feature Correlation Analysis",
            feature_correlation.analyze_correlation,
            generate_heatmap=True
        )
        
        # 10. Extreme Heterogeneity
        self.run_experiment(
            "Extreme Heterogeneity",
            extreme_heterogeneity.test_extreme_noniid,
            alphas=[0.05, 0.01],
            label_skews=[0.95, 0.99]
        )
    
    def run_priority_3(self):
        """Run additional analyses (nice-to-have)."""
        print("\n" + "📊"*40)
        print("PRIORITY 3: ADDITIONAL ANALYSES")
        print("📊"*40 + "\n")
        
        # 11. Statistical Significance
        self.run_experiment(
            "Statistical Significance Tests",
            statistical_tests.run_statistical_tests,
            test_type='t-test',
            alpha=0.05
        )
        
        # 12. Preprocessing Documentation
        self.run_experiment(
            "Preprocessing Documentation",
            preprocessing_docs.document_preprocessing
        )
        
        # 13. Comprehensive Visualizations
        all_data = {name: exp['result'] for name, exp in self.results['experiments'].items() 
                    if exp['status'] == 'success' and exp.get('result')}
        
        self.run_experiment(
            "Comprehensive Visualizations",
            visualization_suite.create_all_plots,
            all_results=all_data
        )
    
    def save_results(self):
        """Save all experiment results."""
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_duration'] = (
            datetime.fromisoformat(self.results['end_time']) - 
            datetime.fromisoformat(self.results['start_time'])
        ).total_seconds()
        
        results_file = self.session_dir / 'master_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"📝 FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total Experiments: {len(self.results['experiments'])}")
        print(f"Successful: {sum(1 for e in self.results['experiments'].values() if e['status'] == 'success')}")
        print(f"Failed: {sum(1 for e in self.results['experiments'].values() if e['status'] == 'failed')}")
        print(f"Total Duration: {self.results['total_duration']:.2f}s ({self.results['total_duration']/60:.2f} minutes)")
        print(f"📁 Results saved to: {results_file}")
        print(f"{'='*80}\n")
        
        return results_file


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive experiment suite")
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--priority', type=int, choices=[1, 2, 3], help='Run specific priority level')
    parser.add_argument('--quick', action='store_true', help='Run quick subset for testing')
    parser.add_argument('--output-dir', default='experiments/results', help='Output directory')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    if args.quick:
        print("🏃 QUICK TEST MODE")
        runner.run_experiment(
            "Quick Ablation Test",
            ablation_study.run_ablation_analysis,
            features_to_test=['vae', 'shapley'],
            quick_mode=True
        )
    elif args.priority == 1:
        runner.run_priority_1()
    elif args.priority == 2:
        runner.run_priority_1()
        runner.run_priority_2()
    elif args.priority == 3 or args.all:
        runner.run_priority_1()
        runner.run_priority_2()
        runner.run_priority_3()
    else:
        # Default: run priority 1
        runner.run_priority_1()
    
    results_file = runner.save_results()
    
    print(f"\n✅ ALL EXPERIMENTS COMPLETED!")
    print(f"📦 Results package: {runner.session_dir}")
    print(f"\nTo run all experiments: python experiments/run_all_experiments.py --all")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

