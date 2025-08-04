#!/usr/bin/env python3
"""
FLTrust Experiments Runner for Paper Comparison

This script runs all necessary FLTrust experiments to generate results
that can be compared with the user's paper results.

Experiments include:
- 3 Datasets: MNIST, CIFAR-10, Alzheimer MRI
- 3 Distribution types: IID, Label-Skew (70%, 90%), Dirichlet (α=0.5, α=0.1)  
- 5 Attacks: Scaling (×10), Partial Scaling (50% dims ×5), Sign Flipping, Gaussian Noise, Label Flipping
- Configuration: 10 clients, 25-30 rounds, 5-8 local epochs, 30% malicious clients
"""

import sys
import os
import time
import json
import csv
from datetime import datetime
import logging

# Import the FLTrust implementation
from fltrust_comprehensive_implementation import (
    FLTrustConfig, FLTrustExperiment, 
    CNNModel, ResNet18Model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fltrust_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FLTrustPaperComparison:
    """
    Runs FLTrust experiments matching the paper's experimental setup
    """
    
    def __init__(self):
        self.config = FLTrustConfig()
        # Match the paper's configuration
        self.config.num_clients = 10
        self.config.global_rounds = 25  # Can be adjusted to 30 if needed
        self.config.local_epochs = 5    # Can be adjusted to 8 if needed
        self.config.malicious_fraction = 0.3  # 30% malicious clients
        self.config.root_dataset_size = 100
        
        self.experiment = FLTrustExperiment(self.config)
        self.results = {}
        
    def run_all_experiments(self):
        """Run all experiments needed for paper comparison"""
        
        logger.info("="*80)
        logger.info("Starting FLTrust Comprehensive Experiments for Paper Comparison")
        logger.info("="*80)
        
        # Define experimental parameters matching the paper
        datasets = [
            {"name": "mnist", "description": "MNIST (CNN)"},
            {"name": "cifar10", "description": "CIFAR-10 (ResNet-18)"},
            {"name": "alzheimer", "description": "Alzheimer MRI (ResNet-18)"}
        ]
        
        distributions = [
            {"type": "iid", "params": {}, "description": "IID"},
            {"type": "label_skew", "params": {"skew_ratio": 0.7}, "description": "Label-Skew (70%)"},
            {"type": "label_skew", "params": {"skew_ratio": 0.9}, "description": "Label-Skew (90%)"},
            {"type": "dirichlet", "params": {"alpha": 0.5}, "description": "Dirichlet (α=0.5)"},
            {"type": "dirichlet", "params": {"alpha": 0.1}, "description": "Dirichlet (α=0.1)"}
        ]
        
        attacks = [
            {"name": "clean", "description": "No Attack (Clean)"},
            {"name": "scaling", "description": "Scaling Attack (×10)"},
            {"name": "partial_scaling", "description": "Partial Scaling (50% dims ×5)"},
            {"name": "sign_flipping", "description": "Sign Flipping Attack"},
            {"name": "gaussian_noise", "description": "Gaussian Noise Attack"},
            {"name": "label_flipping", "description": "Label Flipping (50%)"}
        ]
        
        total_experiments = len(datasets) * len(distributions) * len(attacks)
        experiment_count = 0
        
        logger.info(f"Total experiments to run: {total_experiments}")
        logger.info(f"Configuration: {self.config.num_clients} clients, "
                   f"{self.config.global_rounds} rounds, "
                   f"{self.config.local_epochs} local epochs, "
                   f"{self.config.malicious_fraction*100}% malicious clients")
        logger.info("-" * 80)
        
        start_time = time.time()
        
        for dataset_info in datasets:
            for dist_info in distributions:
                for attack_info in attacks:
                    experiment_count += 1
                    
                    dataset_name = dataset_info["name"]
                    dist_type = dist_info["type"] 
                    attack_type = attack_info["name"]
                    
                    logger.info(f"\nExperiment {experiment_count}/{total_experiments}")
                    logger.info(f"Dataset: {dataset_info['description']}")
                    logger.info(f"Distribution: {dist_info['description']}")
                    logger.info(f"Attack: {attack_info['description']}")
                    logger.info("-" * 50)
                    
                    exp_start_time = time.time()
                    
                    try:
                        # Run experiment
                        result = self.experiment.run_experiment(
                            dataset_name=dataset_name,
                            distribution_type=dist_type,
                            attack_type=attack_type,
                            dataset_params=dist_info["params"]
                        )
                        
                        # Store result with comprehensive key
                        result_key = f"{dataset_name}_{dist_type}"
                        if dist_info["params"]:
                            param_str = "_".join([f"{k}{v}" for k, v in dist_info["params"].items()])
                            result_key += f"_{param_str}"
                        result_key += f"_{attack_type}"
                        
                        # Add experiment metadata
                        result["experiment_info"] = {
                            "dataset_description": dataset_info["description"],
                            "distribution_description": dist_info["description"],
                            "attack_description": attack_info["description"],
                            "experiment_number": experiment_count,
                            "total_experiments": total_experiments
                        }
                        
                        self.results[result_key] = result
                        
                        exp_duration = time.time() - exp_start_time
                        logger.info(f"✓ Completed in {exp_duration:.1f}s - "
                                   f"Final Accuracy: {result['final_accuracy']:.2f}%")
                        
                        # Save intermediate results
                        self.save_results_incremental()
                        
                    except Exception as e:
                        logger.error(f"✗ Failed: {str(e)}")
                        continue
        
        total_duration = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"All experiments completed in {total_duration/60:.1f} minutes")
        logger.info(f"Total successful experiments: {len(self.results)}")
        logger.info(f"{'='*80}")
        
        # Generate final analysis
        self.generate_comprehensive_analysis()
        
        return self.results
    
    def save_results_incremental(self):
        """Save results incrementally to avoid data loss"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fltrust_results_{timestamp}.json"
        self.experiment.save_results(self.results, filename)
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis and comparison tables"""
        
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPREHENSIVE ANALYSIS")
        logger.info("="*80)
        
        # 1. Generate accuracy comparison table
        self.generate_accuracy_table()
        
        # 2. Generate attack robustness analysis
        self.generate_attack_analysis()
        
        # 3. Generate distribution analysis
        self.generate_distribution_analysis()
        
        # 4. Generate CSV for easy import to paper
        self.generate_csv_results()
        
        # 5. Generate LaTeX table for paper
        self.generate_latex_table()
    
    def generate_accuracy_table(self):
        """Generate accuracy comparison table"""
        
        logger.info("\n" + "-"*80)
        logger.info("FLTRUST ACCURACY RESULTS SUMMARY")
        logger.info("-"*80)
        
        # Organize results by dataset and distribution
        table_data = {}
        
        for key, result in self.results.items():
            parts = key.split('_')
            dataset = parts[0]
            distribution = "_".join(parts[1:-1])  # Handle complex distribution names
            attack = parts[-1]
            
            if dataset not in table_data:
                table_data[dataset] = {}
            if distribution not in table_data[dataset]:
                table_data[dataset][distribution] = {}
            
            table_data[dataset][distribution][attack] = result['final_accuracy']
        
        # Print formatted table
        attacks = ["clean", "scaling", "partial_scaling", "sign_flipping", "gaussian_noise", "label_flipping"]
        
        for dataset in sorted(table_data.keys()):
            logger.info(f"\n📊 DATASET: {dataset.upper()}")
            logger.info("-" * 70)
            
            # Headers
            header = f"{'Distribution':<20} " + " ".join([f"{att[:8]:<10}" for att in attacks]) + f" {'Avg':<8}"
            logger.info(header)
            logger.info("-" * len(header))
            
            for distribution in sorted(table_data[dataset].keys()):
                row_data = []
                accuracies = []
                
                for attack in attacks:
                    if attack in table_data[dataset][distribution]:
                        acc = table_data[dataset][distribution][attack]
                        row_data.append(f"{acc:<10.2f}")
                        accuracies.append(acc)
                    else:
                        row_data.append(f"{'N/A':<10}")
                
                avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
                row = f"{distribution:<20} " + " ".join(row_data) + f" {avg_acc:<8.2f}"
                logger.info(row)
    
    def generate_attack_analysis(self):
        """Generate attack-specific analysis"""
        
        logger.info("\n" + "-"*60)
        logger.info("ATTACK ROBUSTNESS ANALYSIS")
        logger.info("-"*60)
        
        attack_performance = {}
        
        for key, result in self.results.items():
            parts = key.split('_')
            attack = parts[-1]
            
            if attack not in attack_performance:
                attack_performance[attack] = []
            
            attack_performance[attack].append(result['final_accuracy'])
        
        # Calculate statistics for each attack
        for attack in sorted(attack_performance.keys()):
            accuracies = attack_performance[attack]
            avg_acc = sum(accuracies) / len(accuracies)
            min_acc = min(accuracies)
            max_acc = max(accuracies)
            
            logger.info(f"{attack.upper():<15}: "
                       f"Avg={avg_acc:6.2f}% | "
                       f"Min={min_acc:6.2f}% | "
                       f"Max={max_acc:6.2f}% | "
                       f"Samples={len(accuracies)}")
    
    def generate_distribution_analysis(self):
        """Generate distribution-specific analysis"""
        
        logger.info("\n" + "-"*60)
        logger.info("DISTRIBUTION IMPACT ANALYSIS")
        logger.info("-"*60)
        
        dist_performance = {}
        
        for key, result in self.results.items():
            parts = key.split('_')
            # Extract distribution (handle complex names)
            if "alpha" in key:
                if "alpha0.5" in key:
                    dist = "dirichlet_α0.5"
                elif "alpha0.1" in key:
                    dist = "dirichlet_α0.1"
                else:
                    dist = "dirichlet"
            elif "skew_ratio" in key:
                if "skew_ratio0.7" in key:
                    dist = "label_skew_70%"
                elif "skew_ratio0.9" in key:
                    dist = "label_skew_90%"
                else:
                    dist = "label_skew"
            else:
                dist = parts[1]
            
            if dist not in dist_performance:
                dist_performance[dist] = []
            
            dist_performance[dist].append(result['final_accuracy'])
        
        # Calculate statistics for each distribution
        for dist in sorted(dist_performance.keys()):
            accuracies = dist_performance[dist]
            avg_acc = sum(accuracies) / len(accuracies)
            min_acc = min(accuracies)
            max_acc = max(accuracies)
            
            logger.info(f"{dist:<15}: "
                       f"Avg={avg_acc:6.2f}% | "
                       f"Min={min_acc:6.2f}% | "
                       f"Max={max_acc:6.2f}% | "
                       f"Samples={len(accuracies)}")
    
    def generate_csv_results(self):
        """Generate CSV file for easy import to papers/analysis"""
        
        csv_data = []
        
        for key, result in self.results.items():
            parts = key.split('_')
            dataset = parts[0]
            attack = parts[-1]
            
            # Parse distribution
            if "alpha" in key:
                if "alpha0.5" in key:
                    distribution = "Dirichlet (α=0.5)"
                elif "alpha0.1" in key:
                    distribution = "Dirichlet (α=0.1)"
                else:
                    distribution = "Dirichlet"
            elif "skew_ratio" in key:
                if "skew_ratio0.7" in key:
                    distribution = "Label-Skew (70%)"
                elif "skew_ratio0.9" in key:
                    distribution = "Label-Skew (90%)"
                else:
                    distribution = "Label-Skew"
            else:
                distribution = "IID"
            
            csv_data.append({
                'Dataset': dataset.upper(),
                'Distribution': distribution,
                'Attack': attack.replace('_', ' ').title(),
                'Accuracy': result['final_accuracy'],
                'Loss': result['final_loss'],
                'Malicious_Clients': len(result['malicious_clients']),
                'Total_Rounds': result['config']['global_rounds'],
                'Local_Epochs': result['config']['local_epochs']
            })
        
        # Save CSV
        csv_filename = f"fltrust_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            if csv_data:
                fieldnames = csv_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"\n📁 CSV results saved to: {csv_filename}")
        
        # Print summary statistics
        logger.info("\n" + "-"*50)
        logger.info("SUMMARY STATISTICS")
        logger.info("-"*50)
        
        accuracies = [row['Accuracy'] for row in csv_data]
        if accuracies:
            overall_avg = sum(accuracies) / len(accuracies)
            overall_min = min(accuracies)
            overall_max = max(accuracies)
            
            # Calculate standard deviation manually
            variance = sum((x - overall_avg) ** 2 for x in accuracies) / len(accuracies)
            overall_std = variance ** 0.5
            
            logger.info(f"Overall Average Accuracy: {overall_avg:.2f}% (±{overall_std:.2f})")
            logger.info(f"Range: {overall_min:.2f}% - {overall_max:.2f}%")
        
        logger.info(f"Total Experiments: {len(csv_data)}")
    
    def generate_latex_table(self):
        """Generate LaTeX table for direct use in paper"""
        
        logger.info("\n" + "-"*60)
        logger.info("LATEX TABLE FOR PAPER")
        logger.info("-"*60)
        
        # Organize data for LaTeX table
        latex_data = {}
        
        for key, result in self.results.items():
            parts = key.split('_')
            dataset = parts[0]
            attack = parts[-1]
            
            # Parse distribution
            if "alpha0.5" in key:
                dist = "Dirichlet (α=0.5)"
            elif "alpha0.1" in key:
                dist = "Dirichlet (α=0.1)"
            elif "skew_ratio0.7" in key:
                dist = "Label-Skew (70%)"
            elif "skew_ratio0.9" in key:
                dist = "Label-Skew (90%)"
            elif "iid" in key:
                dist = "IID"
            else:
                dist = "Other"
            
            if dataset not in latex_data:
                latex_data[dataset] = {}
            if dist not in latex_data[dataset]:
                latex_data[dataset][dist] = {}
            
            latex_data[dataset][dist][attack] = result['final_accuracy']
        
        # Generate LaTeX table
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{FLTrust Performance Comparison}\n"
        latex_table += "\\label{tab:fltrust_results}\n"
        latex_table += "\\begin{tabular}{|l|l|c|c|c|c|c|c|}\n"
        latex_table += "\\hline\n"
        latex_table += "\\textbf{Dataset} & \\textbf{Distribution} & \\textbf{Clean} & \\textbf{Scaling} & \\textbf{Partial} & \\textbf{Sign Flip} & \\textbf{Noise} & \\textbf{Label Flip} \\\\\n"
        latex_table += "\\hline\n"
        
        attacks = ["clean", "scaling", "partial_scaling", "sign_flipping", "gaussian_noise", "label_flipping"]
        
        for dataset in sorted(latex_data.keys()):
            first_row = True
            for dist in sorted(latex_data[dataset].keys()):
                if first_row:
                    dataset_cell = f"\\multirow{{{len(latex_data[dataset])}}}{{*}}{{{dataset.upper()}}}"
                    first_row = False
                else:
                    dataset_cell = ""
                
                row = f"{dataset_cell} & {dist}"
                
                for attack in attacks:
                    if attack in latex_data[dataset][dist]:
                        acc = latex_data[dataset][dist][attack]
                        row += f" & {acc:.1f}\\%"
                    else:
                        row += " & N/A"
                
                row += " \\\\\n"
                latex_table += row
            
            latex_table += "\\hline\n"
        
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"
        
        # Save LaTeX table
        latex_filename = f"fltrust_latex_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        with open(latex_filename, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"📄 LaTeX table saved to: {latex_filename}")
        logger.info("\nLaTeX Table Preview:")
        logger.info("-" * 60)
        print(latex_table)

def main():
    """Main function to run all FLTrust experiments"""
    
    print("🚀 FLTrust Comprehensive Experiments for Paper Comparison")
    print("=" * 80)
    print("This will run all experiments needed to compare FLTrust with your paper results.")
    print("Estimated time: 2-4 hours depending on your hardware.")
    print("Results will be saved incrementally to avoid data loss.")
    print("=" * 80)
    
    response = input("\nProceed with experiments? (y/n): ")
    if response.lower() != 'y':
        print("Experiments cancelled.")
        return
    
    # Initialize and run experiments
    comparison = FLTrustPaperComparison()
    
    try:
        results = comparison.run_all_experiments()
        
        print("\n🎉 All experiments completed successfully!")
        print(f"📊 Total experiments: {len(results)}")
        print("📁 Results saved in multiple formats:")
        print("   - JSON: Detailed results with training history")
        print("   - CSV: Summary table for analysis")
        print("   - LaTeX: Ready-to-use table for paper")
        print("   - Log: Detailed execution log")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user.")
        print("📁 Partial results have been saved.")
        
    except Exception as e:
        logger.error(f"❌ Experiments failed: {str(e)}")
        print("📁 Partial results may have been saved.")

if __name__ == "__main__":
    main() 