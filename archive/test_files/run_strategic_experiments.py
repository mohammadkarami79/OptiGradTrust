#!/usr/bin/env python3
"""
Strategic Federated Learning Experiment Runner

This script implements a phased approach to handle large-scale federated learning 
experiments efficiently. Instead of running 540+ experiments, it uses strategic
sampling to get research-quality results with manageable computational requirements.

Key Features:
- Phased execution (Core -> Extended -> Comprehensive)
- Progress tracking with recovery
- Automated result analysis and comparison
- Publication-quality plotting
- Smart resource management

Usage:
    python run_strategic_experiments.py --phase core
    python run_strategic_experiments.py --phase extended  
    python run_strategic_experiments.py --continue
    python run_strategic_experiments.py --analyze
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class StrategicExperimentManager:
    """Manages strategic federated learning experiments with phased execution."""
    
    def __init__(self, config_file: str = "experiment_configs_strategic.json"):
        self.config_file = config_file
        self.results_dir = "results/strategic_experiments"
        self.progress_file = os.path.join(self.results_dir, "progress.json")
        self.summary_file = os.path.join(self.results_dir, "experiment_summary.json")
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs("results/configs", exist_ok=True)
        os.makedirs("results/metrics", exist_ok=True)
        os.makedirs("research_plots", exist_ok=True)
        
        # Load strategic configuration
        self.config = self._load_strategic_config()
        
        # Initialize progress tracking
        self.progress = self._load_progress()
        
    def _load_strategic_config(self) -> Dict[str, Any]:
        """Load strategic experiment configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded strategic configuration: {self.config_file}")
            return config
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load experiment progress or create new."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                print(f"📊 Loaded existing progress: {len(progress.get('completed', []))} experiments completed")
                return progress
            except:
                pass
        
        # Create new progress tracking
        return {
            'started_at': datetime.now().isoformat(),
            'completed': [],
            'failed': [],
            'current_phase': None,
            'total_time_hours': 0.0
        }
    
    def _save_progress(self):
        """Save current progress."""
        self.progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2, default=str)
    
    def run_phase(self, phase_name: str):
        """Run a specific phase of experiments."""
        
        if phase_name not in self.config:
            print(f"❌ Phase '{phase_name}' not found in configuration")
            return
        
        phase_config = self.config[phase_name]
        experiments = phase_config.get('experiments', [])
        
        print(f"\n🚀 STARTING PHASE: {phase_name.upper()}")
        print(f"📊 Total experiments: {len(experiments)}")
        print(f"⏱️  Estimated time: {phase_config.get('estimated_time_hours', 'unknown')} hours")
        print(f"📝 Description: {phase_config.get('description', 'No description')}")
        
        # Confirm execution for large phases
        if len(experiments) > 10:
            confirm = input(f"\n⚠️  This will run {len(experiments)} experiments. Continue? (yes/no): ")
            if confirm.lower() != 'yes':
                print("❌ Execution cancelled")
                return
        
        # Update progress
        self.progress['current_phase'] = phase_name
        self._save_progress()
        
        # Run experiments
        phase_start_time = time.time()
        phase_results = []
        
        for i, experiment in enumerate(experiments):
            print(f"\n{'='*80}")
            print(f"🧪 EXPERIMENT {i+1}/{len(experiments)}: {experiment['name']}")
            print(f"⏱️  Progress: {len(self.progress['completed'])}/{len(experiments)} phase experiments completed")
            
            try:
                # Check if experiment already completed
                if experiment['name'] in [exp['name'] for exp in self.progress['completed']]:
                    print(f"⏭️  Skipping: Already completed")
                    continue
                
                # Run single experiment
                result = self._run_single_experiment(experiment)
                
                # Add to results
                phase_results.append(result)
                self.progress['completed'].append(result)
                
                # Save progress after each experiment
                self._save_progress()
                
                print(f"✅ Completed: {experiment['name']}")
                
            except Exception as e:
                print(f"❌ Failed: {experiment['name']}")
                print(f"   Error: {str(e)}")
                
                error_result = experiment.copy()
                error_result.update({
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                self.progress['failed'].append(error_result)
                self._save_progress()
        
        # Phase completion
        phase_time = (time.time() - phase_start_time) / 3600
        self.progress['total_time_hours'] += phase_time
        self.progress['current_phase'] = None
        self._save_progress()
        
        print(f"\n🎉 PHASE COMPLETED: {phase_name}")
        print(f"⏱️  Phase time: {phase_time:.1f} hours")
        print(f"⏱️  Total time: {self.progress['total_time_hours']:.1f} hours")
        print(f"✅ Successful: {len(phase_results)}")
        print(f"❌ Failed: {len([exp for exp in self.progress['failed'] if 'phase' in exp.get('name', '')])}")
        
        # Create phase summary
        self._create_phase_summary(phase_name, phase_results)
    
    def _run_single_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment by updating config and executing main.py."""
        
        print(f"📝 Configuring experiment...")
        
        # Update federated learning configuration
        config_updates = self._prepare_config_updates(experiment)
        
        # Apply configuration updates
        self._apply_config_updates(config_updates)
        
        # Run the experiment
        print(f"🚀 Starting federated learning...")
        experiment_start_time = time.time()
        
        try:
            # Import and run main function
            if 'main' in sys.modules:
                del sys.modules['main']
            
            from main import main
            result = main()
            
            # Process and enhance result
            experiment_time = (time.time() - experiment_start_time) / 60
            
            enhanced_result = {
                **result,
                'experiment_config': experiment,
                'execution_time_minutes': experiment_time,
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'experiment_name': experiment['name']
            }
            
            print(f"✅ Experiment completed in {experiment_time:.1f} minutes")
            
            return enhanced_result
            
        except Exception as e:
            print(f"❌ Experiment failed: {str(e)}")
            raise e
    
    def _prepare_config_updates(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration updates for the experiment."""
        
        # Core configuration mapping
        config_updates = {
            'DATASET': experiment.get('DATASET', 'MNIST'),
            'MODEL': experiment.get('MODEL', 'CNN'),
            'AGGREGATION_METHOD': experiment.get('AGGREGATION_METHOD', 'fedbn'),
            'ATTACK_TYPE': experiment.get('ATTACK_TYPE', 'partial_scaling_attack'),
            'NUM_CLIENTS': experiment.get('NUM_CLIENTS', 10),
            'FRACTION_MALICIOUS': experiment.get('FRACTION_MALICIOUS', 0.3),
            'GLOBAL_EPOCHS': experiment.get('GLOBAL_EPOCHS', 30),
            'LOCAL_EPOCHS_CLIENT': experiment.get('LOCAL_EPOCHS_CLIENT', 10),
        }
        
        # Data distribution configuration
        if 'DATA_DISTRIBUTION' in experiment:
            config_updates['DATA_DISTRIBUTION'] = experiment['DATA_DISTRIBUTION']
        
        if 'ENABLE_NON_IID' in experiment:
            config_updates['ENABLE_NON_IID'] = experiment['ENABLE_NON_IID']
        
        if 'DIRICHLET_ALPHA' in experiment:
            config_updates['DIRICHLET_ALPHA'] = experiment['DIRICHLET_ALPHA']
        
        # Attack-specific parameters
        if 'SCALING_FACTOR' in experiment:
            config_updates['SCALING_FACTOR'] = experiment['SCALING_FACTOR']
        
        if 'PARTIAL_SCALING_PERCENT' in experiment:
            config_updates['PARTIAL_SCALING_PERCENT'] = experiment['PARTIAL_SCALING_PERCENT']
        
        if 'NOISE_FACTOR' in experiment:
            config_updates['NOISE_FACTOR'] = experiment['NOISE_FACTOR']
        
        return config_updates
    
    def _apply_config_updates(self, config_updates: Dict[str, Any]):
        """Apply configuration updates to the federated learning system."""
        
        # Write configuration overrides
        override_file = 'temp_config_override.py'
        with open(override_file, 'w') as f:
            f.write("# Temporary configuration overrides for strategic experiments\n")
            f.write("# This file is auto-generated and will be imported by config.py\n\n")
            
            for key, value in config_updates.items():
                if isinstance(value, str):
                    f.write(f"{key} = '{value}'\n")
                elif isinstance(value, bool):
                    f.write(f"{key} = {value}\n")
                else:
                    f.write(f"{key} = {value}\n")
        
        print(f"📝 Applied {len(config_updates)} configuration updates")
    
    def _create_phase_summary(self, phase_name: str, results: List[Dict[str, Any]]):
        """Create comprehensive summary for completed phase."""
        
        if not results:
            print("⚠️  No results to summarize")
            return
        
        # Calculate summary statistics
        successful_results = [r for r in results if r.get('status') == 'completed']
        
        if not successful_results:
            print("⚠️  No successful results to analyze")
            return
        
        # Extract key metrics
        final_accuracies = [r.get('final_accuracy', 0) for r in successful_results]
        initial_accuracies = [r.get('initial_accuracy', 0) for r in successful_results]
        improvements = [f - i for f, i in zip(final_accuracies, initial_accuracies)]
        
        detection_precisions = []
        detection_recalls = []
        detection_f1s = []
        
        for r in successful_results:
            det_metrics = r.get('detection_metrics', {})
            detection_precisions.append(det_metrics.get('precision', 0))
            detection_recalls.append(det_metrics.get('recall', 0))
            detection_f1s.append(det_metrics.get('f1_score', 0))
        
        # Create summary
        summary = {
            'phase_name': phase_name,
            'completed_at': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'completion_rate': len(successful_results) / len(results) if results else 0,
            
            # Accuracy metrics
            'accuracy_stats': {
                'mean_final_accuracy': np.mean(final_accuracies),
                'std_final_accuracy': np.std(final_accuracies),
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'best_improvement': max(improvements) if improvements else 0,
                'worst_improvement': min(improvements) if improvements else 0
            },
            
            # Detection metrics
            'detection_stats': {
                'mean_precision': np.mean(detection_precisions),
                'std_precision': np.std(detection_precisions),
                'mean_recall': np.mean(detection_recalls),
                'std_recall': np.std(detection_recalls),
                'mean_f1_score': np.mean(detection_f1s),
                'std_f1_score': np.std(detection_f1s)
            },
            
            # Execution metrics
            'execution_stats': {
                'total_time_hours': self.progress['total_time_hours'],
                'average_experiment_time': np.mean([r.get('execution_time_minutes', 0) for r in successful_results]),
                'experiments_per_hour': len(successful_results) / self.progress['total_time_hours'] if self.progress['total_time_hours'] > 0 else 0
            }
        }
        
        # Save summary
        summary_path = os.path.join(self.results_dir, f"{phase_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 PHASE SUMMARY: {phase_name}")
        print(f"{'='*60}")
        print(f"✅ Success Rate: {summary['completion_rate']:.1%}")
        print(f"📈 Mean Accuracy Improvement: {summary['accuracy_stats']['mean_improvement']:.4f} ± {summary['accuracy_stats']['std_improvement']:.4f}")
        print(f"🎯 Mean Detection Precision: {summary['detection_stats']['mean_precision']:.3f} ± {summary['detection_stats']['std_precision']:.3f}")
        print(f"🎯 Mean Detection Recall: {summary['detection_stats']['mean_recall']:.3f} ± {summary['detection_stats']['std_recall']:.3f}")
        print(f"⏱️  Total Time: {summary['execution_stats']['total_time_hours']:.1f} hours")
        print(f"📁 Summary saved: {summary_path}")
        
        # Create phase comparison plots
        self._create_phase_plots(phase_name, successful_results, summary)
    
    def _create_phase_plots(self, phase_name: str, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Create comprehensive phase analysis plots."""
        
        if len(results) < 2:
            print("⚠️  Insufficient results for plotting")
            return
        
        # Prepare data
        experiment_names = [r['experiment_name'] for r in results]
        final_accuracies = [r.get('final_accuracy', 0) for r in results]
        improvements = [r.get('final_accuracy', 0) - r.get('initial_accuracy', 0) for r in results]
        detection_precisions = [r.get('detection_metrics', {}).get('precision', 0) for r in results]
        detection_f1s = [r.get('detection_metrics', {}).get('f1_score', 0) for r in results]
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Strategic Experiment Analysis: {phase_name.upper()}', fontsize=16, fontweight='bold')
        
        # Plot 1: Final Accuracy by Experiment
        axes[0, 0].bar(range(len(results)), final_accuracies, alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Final Accuracy by Experiment')
        axes[0, 0].set_ylabel('Final Accuracy')
        axes[0, 0].set_xticks(range(len(results)))
        axes[0, 0].set_xticklabels([name.split('_')[-3:] for name in experiment_names], rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Accuracy Improvement Distribution
        axes[0, 1].hist(improvements, bins=min(10, len(improvements)), alpha=0.7, color='forestgreen', edgecolor='black')
        axes[0, 1].axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(improvements):.4f}')
        axes[0, 1].set_title('Accuracy Improvement Distribution')
        axes[0, 1].set_xlabel('Accuracy Improvement')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Detection Performance
        x_pos = np.arange(len(results))
        width = 0.35
        axes[1, 0].bar(x_pos - width/2, detection_precisions, width, alpha=0.7, color='orange', label='Precision')
        axes[1, 0].bar(x_pos + width/2, detection_f1s, width, alpha=0.7, color='purple', label='F1-Score')
        axes[1, 0].set_title('Detection Performance by Experiment')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([name.split('_')[-2:] for name in experiment_names], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Performance vs Detection Trade-off
        axes[1, 1].scatter(improvements, detection_precisions, alpha=0.7, s=100, c='red')
        axes[1, 1].set_title('Accuracy Improvement vs Detection Precision')
        axes[1, 1].set_xlabel('Accuracy Improvement')
        axes[1, 1].set_ylabel('Detection Precision')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(improvements) > 1:
            z = np.polyfit(improvements, detection_precisions, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(sorted(improvements), p(sorted(improvements)), "b--", alpha=0.8, label='Trend')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join("research_plots", f"{phase_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Phase analysis plot saved: {plot_path}")
    
    def analyze_all_results(self):
        """Create comprehensive analysis of all completed experiments."""
        
        print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        if not self.progress['completed']:
            print("❌ No completed experiments to analyze")
            return
        
        successful_results = [r for r in self.progress['completed'] if r.get('status') == 'completed']
        
        if not successful_results:
            print("❌ No successful experiments to analyze")
            return
        
        print(f"📈 Total completed experiments: {len(successful_results)}")
        print(f"⏱️  Total experiment time: {self.progress['total_time_hours']:.1f} hours")
        
        # Create comprehensive comparison
        self._create_comprehensive_comparison(successful_results)
        
        # Create summary report
        self._create_final_summary_report(successful_results)
    
    def _create_comprehensive_comparison(self, results: List[Dict[str, Any]]):
        """Create comprehensive comparison across all experiments."""
        
        # Prepare data for analysis
        df_data = []
        for result in results:
            experiment_config = result.get('experiment_config', {})
            detection_metrics = result.get('detection_metrics', {})
            
            row = {
                'experiment_name': result.get('experiment_name', 'Unknown'),
                'dataset': experiment_config.get('DATASET', 'Unknown'),
                'aggregation': experiment_config.get('AGGREGATION_METHOD', 'Unknown'),
                'attack_type': experiment_config.get('ATTACK_TYPE', 'Unknown'),
                'data_distribution': experiment_config.get('DATA_DISTRIBUTION', 'Unknown'),
                'final_accuracy': result.get('final_accuracy', 0),
                'initial_accuracy': result.get('initial_accuracy', 0),
                'improvement': result.get('final_accuracy', 0) - result.get('initial_accuracy', 0),
                'detection_precision': detection_metrics.get('precision', 0),
                'detection_recall': detection_metrics.get('recall', 0),
                'detection_f1': detection_metrics.get('f1_score', 0),
                'execution_time': result.get('execution_time_minutes', 0)
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save comprehensive results
        results_csv_path = os.path.join(self.results_dir, "comprehensive_results.csv")
        df.to_csv(results_csv_path, index=False)
        print(f"📁 Comprehensive results saved: {results_csv_path}")
        
        # Create analysis by category
        self._analyze_by_category(df)
    
    def _analyze_by_category(self, df: pd.DataFrame):
        """Analyze results by different categories."""
        
        print(f"\n📊 ANALYSIS BY CATEGORY")
        print(f"{'='*40}")
        
        # Analysis by aggregation method
        if 'aggregation' in df.columns:
            print(f"\n🔧 BY AGGREGATION METHOD:")
            agg_analysis = df.groupby('aggregation').agg({
                'improvement': ['mean', 'std', 'count'],
                'detection_precision': ['mean', 'std'],
                'detection_f1': ['mean', 'std']
            }).round(4)
            print(agg_analysis)
        
        # Analysis by attack type
        if 'attack_type' in df.columns:
            print(f"\n⚔️  BY ATTACK TYPE:")
            attack_analysis = df.groupby('attack_type').agg({
                'improvement': ['mean', 'std', 'count'],
                'detection_precision': ['mean', 'std'],
                'detection_f1': ['mean', 'std']
            }).round(4)
            print(attack_analysis)
        
        # Analysis by data distribution
        if 'data_distribution' in df.columns:
            print(f"\n📊 BY DATA DISTRIBUTION:")
            dist_analysis = df.groupby('data_distribution').agg({
                'improvement': ['mean', 'std', 'count'],
                'detection_precision': ['mean', 'std'],
                'detection_f1': ['mean', 'std']
            }).round(4)
            print(dist_analysis)
    
    def _create_final_summary_report(self, results: List[Dict[str, Any]]):
        """Create final comprehensive summary report."""
        
        # Overall statistics
        final_accuracies = [r.get('final_accuracy', 0) for r in results]
        improvements = [r.get('final_accuracy', 0) - r.get('initial_accuracy', 0) for r in results]
        detection_precisions = [r.get('detection_metrics', {}).get('precision', 0) for r in results]
        detection_f1s = [r.get('detection_metrics', {}).get('f1_score', 0) for r in results]
        
        final_report = {
            'report_generated': datetime.now().isoformat(),
            'total_experiments': len(results),
            'total_time_hours': self.progress['total_time_hours'],
            
            'overall_performance': {
                'mean_final_accuracy': float(np.mean(final_accuracies)),
                'mean_improvement': float(np.mean(improvements)),
                'best_improvement': float(max(improvements)) if improvements else 0,
                'worst_improvement': float(min(improvements)) if improvements else 0,
                'positive_improvements': int(sum(1 for imp in improvements if imp > 0)),
                'improvement_success_rate': float(sum(1 for imp in improvements if imp > 0) / len(improvements)) if improvements else 0
            },
            
            'detection_performance': {
                'mean_precision': float(np.mean(detection_precisions)),
                'mean_f1_score': float(np.mean(detection_f1s)),
                'best_precision': float(max(detection_precisions)) if detection_precisions else 0,
                'experiments_above_50_precision': int(sum(1 for p in detection_precisions if p > 0.5))
            },
            
            'efficiency_metrics': {
                'experiments_per_hour': float(len(results) / self.progress['total_time_hours']) if self.progress['total_time_hours'] > 0 else 0,
                'estimated_full_factorial_time': 540 * (self.progress['total_time_hours'] / len(results)) if len(results) > 0 else 0,
                'time_savings_percent': (1 - len(results) / 540) * 100 if len(results) < 540 else 0
            },
            
            'research_recommendations': self._generate_research_recommendations(results)
        }
        
        # Save final report
        report_path = os.path.join(self.results_dir, "final_research_report.json")
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Print executive summary
        print(f"\n🎯 EXECUTIVE SUMMARY")
        print(f"{'='*50}")
        print(f"📊 Total Experiments: {final_report['total_experiments']}")
        print(f"⏱️  Total Time: {final_report['total_time_hours']:.1f} hours")
        print(f"📈 Mean Accuracy Improvement: {final_report['overall_performance']['mean_improvement']:.4f}")
        print(f"🎯 Mean Detection Precision: {final_report['detection_performance']['mean_precision']:.3f}")
        print(f"⚡ Time Savings vs Full Factorial: {final_report['efficiency_metrics']['time_savings_percent']:.1f}%")
        print(f"📁 Full report saved: {report_path}")
    
    def _generate_research_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate research recommendations based on results."""
        
        recommendations = []
        
        # Analyze improvements
        improvements = [r.get('final_accuracy', 0) - r.get('initial_accuracy', 0) for r in results]
        positive_improvements = [imp for imp in improvements if imp > 0]
        
        if len(positive_improvements) / len(improvements) > 0.7:
            recommendations.append("Strong positive model improvement across experiments - system is working well")
        elif len(positive_improvements) / len(improvements) > 0.3:
            recommendations.append("Mixed model improvement results - consider parameter optimization")
        else:
            recommendations.append("Low model improvement rate - recommend threshold adjustment")
        
        # Analyze detection performance
        detection_precisions = [r.get('detection_metrics', {}).get('precision', 0) for r in results]
        high_precision_count = sum(1 for p in detection_precisions if p > 0.7)
        
        if high_precision_count / len(detection_precisions) > 0.5:
            recommendations.append("Excellent detection performance - ready for publication")
        elif high_precision_count / len(detection_precisions) > 0.3:
            recommendations.append("Good detection performance - consider extended experiments")
        else:
            recommendations.append("Detection performance needs improvement - optimize thresholds")
        
        # Time efficiency
        if len(results) < 50:
            recommendations.append("Efficient strategic sampling - suitable for rapid research iteration")
        
        return recommendations

def main():
    """Main function for strategic experiment runner."""
    
    parser = argparse.ArgumentParser(description='Strategic Federated Learning Experiment Runner')
    parser.add_argument('--phase', type=str, choices=['phase_1_core', 'phase_2_extended', 'phase_3_comprehensive'], 
                       help='Phase to run')
    parser.add_argument('--continue', action='store_true', help='Continue from last checkpoint')
    parser.add_argument('--analyze', action='store_true', help='Analyze all completed results')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--config', type=str, default='experiment_configs_strategic.json', 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = StrategicExperimentManager(args.config)
    
    print(f"🧪 STRATEGIC FEDERATED LEARNING EXPERIMENT RUNNER")
    print(f"{'='*60}")
    
    if args.status:
        # Show status
        print(f"📊 Current Status:")
        print(f"   Completed: {len(manager.progress['completed'])} experiments")
        print(f"   Failed: {len(manager.progress['failed'])} experiments")
        print(f"   Total Time: {manager.progress['total_time_hours']:.1f} hours")
        print(f"   Current Phase: {manager.progress.get('current_phase', 'None')}")
        return
    
    if args.analyze:
        # Analyze all results
        manager.analyze_all_results()
        return
    
    if args.phase:
        # Run specific phase
        if args.phase in manager.config:
            manager.run_phase(args.phase)
        else:
            print(f"❌ Phase '{args.phase}' not found in configuration")
            print(f"Available phases: {list(manager.config.keys())}")
    
    elif getattr(args, 'continue', False):
        # Continue from checkpoint
        current_phase = manager.progress.get('current_phase')
        if current_phase and current_phase in manager.config:
            print(f"📍 Continuing from phase: {current_phase}")
            manager.run_phase(current_phase)
        else:
            print(f"❌ No phase to continue. Use --phase to start a new phase.")
    
    else:
        # Show available options
        print(f"📋 Available phases:")
        for phase_name, phase_config in manager.config.items():
            if isinstance(phase_config, dict) and 'experiments' in phase_config:
                exp_count = len(phase_config.get('experiments', []))
                est_time = phase_config.get('estimated_time_hours', 'unknown')
                print(f"   🔬 {phase_name}: {exp_count} experiments (~{est_time} hours)")
        
        print(f"\nUsage:")
        print(f"   python run_strategic_experiments.py --phase phase_1_core")
        print(f"   python run_strategic_experiments.py --analyze")
        print(f"   python run_strategic_experiments.py --status")

if __name__ == "__main__":
    main() 