#!/usr/bin/env python3
"""
Optimal Paper Results Test - Full Parameter Comprehensive Analysis
Designed to achieve the best possible results for research paper submission
"""

import os
import sys
import time
import gc
import torch
import pandas as pd
from datetime import datetime
import logging

# Add the federated learning module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'federated_learning'))

def setup_logging():
    """Setup comprehensive logging for the test"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/optimal_test_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def clear_gpu_memory():
    """Comprehensive GPU memory clearing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_optimal_config():
    """Get optimal configuration for best paper results"""
    config = {
        # Optimal training parameters for maximum accuracy
        'num_rounds': 15,  # Extended for better convergence
        'epochs_per_round': 25,  # Full epochs for optimal learning
        'batch_size': 32,  # Standard batch size for stability
        'learning_rate': 0.01,  # Proven optimal learning rate
        
        # Attack detection parameters
        'num_malicious': 3,  # Standard malicious client count
        'attack_types': [
            'scaling_attack',
            'partial_scaling_attack', 
            'sign_flipping_attack',
            'noise_attack',
            'label_flipping_attack'
        ],
        
        # Model configurations optimized for each dataset
        'dataset_configs': {
            'MNIST': {
                'model': 'CNN',
                'batch_size': 64,  # MNIST can handle larger batches
                'epochs': 20,  # MNIST converges faster
                'expected_accuracy': 99.0  # Target accuracy
            },
            'CIFAR10': {
                'model': 'ResNet18',
                'batch_size': 32,  # Standard for CIFAR-10
                'epochs': 30,  # More epochs for complex data
                'expected_accuracy': 85.0  # Realistic target
            },
            'Alzheimer': {
                'model': 'ResNet18', 
                'batch_size': 16,  # Smaller for medical data
                'epochs': 25,  # Balanced epochs
                'expected_accuracy': 96.0  # Target accuracy
            }
        }
    }
    return config

def run_single_optimal_test(dataset, model, attack_type, config):
    """Run single test with optimal parameters"""
    from federated_learning.training.server import FederatedServer
    import federated_learning.config.config as fl_config
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL TEST: {dataset} + {model} + {attack_type}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Clear memory before test
        clear_gpu_memory()
        
        # Get dataset-specific optimal config
        dataset_config = config['dataset_configs'][dataset]
        
        # Update config for optimal parameters
        fl_config.DATASET = dataset
        fl_config.MODEL = model
        fl_config.GLOBAL_EPOCHS = config['num_rounds']
        fl_config.LOCAL_EPOCHS_CLIENT = dataset_config['epochs']
        fl_config.BATCH_SIZE = dataset_config['batch_size']
        fl_config.LEARNING_RATE = config['learning_rate']
        fl_config.NUM_MALICIOUS = config['num_malicious']
        fl_config.ATTACK_TYPE = attack_type
        
        # Set dataset-specific parameters
        if dataset == 'MNIST':
            fl_config.INPUT_CHANNELS = 1
            fl_config.NUM_CLASSES = 10
        elif dataset == 'CIFAR10':
            fl_config.INPUT_CHANNELS = 3
            fl_config.NUM_CLASSES = 10
        elif dataset == 'Alzheimer':
            fl_config.INPUT_CHANNELS = 3
            fl_config.NUM_CLASSES = 4
        
        print(f"📊 Dataset: {dataset}")
        print(f"🧠 Model: {model}")
        print(f"⚠️  Attack: {attack_type}")
        print(f"🔄 Rounds: {config['num_rounds']}")
        print(f"📈 Epochs/Round: {dataset_config['epochs']}")
        print(f"📦 Batch Size: {dataset_config['batch_size']}")
        print(f"🎯 Target Accuracy: {dataset_config['expected_accuracy']}%")
        
        # Initialize server
        server = FederatedServer()
        print("✅ Server initialized successfully")
        
        # Run federated learning
        print("🚀 Starting federated learning...")
        final_model, metrics = server.train()
        
        # Evaluate final model
        print("📊 Evaluating final model...")
        accuracy = server.evaluate_model(final_model)
        
        # Get attack detection metrics
        detection_metrics = server.get_detection_metrics()
        
        execution_time = time.time() - start_time
        
        # Calculate performance scores
        accuracy_score = accuracy * 100
        precision = detection_metrics.get('precision', 0.0) * 100
        recall = detection_metrics.get('recall', 0.0) * 100
        f1_score = detection_metrics.get('f1_score', 0.0) * 100
        
        print(f"\n📈 RESULTS:")
        print(f"   Accuracy: {accuracy_score:.2f}%")
        print(f"   Precision: {precision:.2f}%") 
        print(f"   Recall: {recall:.2f}%")
        print(f"   F1-Score: {f1_score:.2f}%")
        print(f"   Time: {execution_time/60:.1f} minutes")
        
        # Quality assessment
        quality_score = "EXCELLENT" if accuracy_score >= dataset_config['expected_accuracy'] else "GOOD" if accuracy_score >= dataset_config['expected_accuracy'] * 0.9 else "NEEDS_IMPROVEMENT"
        print(f"   Quality: {quality_score}")
        
        result = {
            'dataset': dataset,
            'model': model,
            'attack_type': attack_type,
            'accuracy': accuracy_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'execution_time': execution_time,
            'quality_score': quality_score,
            'target_accuracy': dataset_config['expected_accuracy'],
            'status': 'SUCCESS'
        }
        
        # Clear memory after test
        del server, final_model
        clear_gpu_memory()
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ ERROR: {str(e)}")
        
        # Clear memory on error
        clear_gpu_memory()
        
        return {
            'dataset': dataset,
            'model': model, 
            'attack_type': attack_type,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'execution_time': execution_time,
            'quality_score': 'FAILED',
            'target_accuracy': config['dataset_configs'][dataset]['expected_accuracy'],
            'status': f'ERROR: {str(e)}'
        }

def run_optimal_comprehensive_test():
    """Run comprehensive test with optimal parameters for best paper results"""
    print("🎯 OPTIMAL PAPER RESULTS TEST")
    print("=" * 80)
    print("🎯 OBJECTIVE: Achieve the best possible results for research paper")
    print("📊 SCOPE: All datasets, all attacks, optimal parameters")
    print("⏱️  DURATION: Estimated 12-18 hours for complete analysis")
    print("=" * 80)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting optimal paper results test")
    
    # Get optimal configuration
    config = get_optimal_config()
    
    # Test combinations
    test_combinations = [
        ('MNIST', 'CNN'),
        ('CIFAR10', 'ResNet18'), 
        ('Alzheimer', 'ResNet18')
    ]
    
    all_results = []
    total_tests = len(test_combinations) * len(config['attack_types'])
    current_test = 0
    
    print(f"📋 Total tests planned: {total_tests}")
    print(f"📝 Log file: {log_file}")
    
    # Progressive execution for best results
    for dataset, model in test_combinations:
        print(f"\n🎯 STARTING DATASET: {dataset}")
        print(f"📊 Expected accuracy target: {config['dataset_configs'][dataset]['expected_accuracy']}%")
        
        dataset_results = []
        
        for attack_type in config['attack_types']:
            current_test += 1
            print(f"\n⏳ Progress: {current_test}/{total_tests}")
            
            # Run optimal test
            result = run_single_optimal_test(dataset, model, attack_type, config)
            dataset_results.append(result)
            all_results.append(result)
            
            # Log result
            logging.info(f"Completed {dataset}-{attack_type}: Acc={result['accuracy']:.2f}%, Quality={result['quality_score']}")
            
            # Show progress
            successful_tests = len([r for r in all_results if r['status'] == 'SUCCESS'])
            print(f"✅ Successful tests so far: {successful_tests}/{current_test}")
        
        # Dataset summary
        dataset_success = [r for r in dataset_results if r['status'] == 'SUCCESS']
        if dataset_success:
            avg_accuracy = sum(r['accuracy'] for r in dataset_success) / len(dataset_success)
            avg_precision = sum(r['precision'] for r in dataset_success) / len(dataset_success)
            print(f"\n📊 {dataset} SUMMARY:")
            print(f"   Average Accuracy: {avg_accuracy:.2f}%")
            print(f"   Average Precision: {avg_precision:.2f}%")
            print(f"   Success Rate: {len(dataset_success)}/{len(config['attack_types'])}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results/final_paper_submission_ready"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = f"{results_dir}/OPTIMAL_PAPER_RESULTS_{timestamp}.csv"
    
    df = pd.DataFrame(all_results)
    df.to_csv(results_file, index=False)
    
    # Generate comprehensive analysis
    analysis_file = f"{results_dir}/OPTIMAL_RESULTS_ANALYSIS_{timestamp}.md"
    generate_optimal_analysis(all_results, analysis_file, config)
    
    print(f"\n🎉 OPTIMAL TEST COMPLETED!")
    print(f"📄 Results saved: {results_file}")
    print(f"📊 Analysis saved: {analysis_file}")
    
    return results_file, analysis_file

def generate_optimal_analysis(results, analysis_file, config):
    """Generate comprehensive analysis of optimal results"""
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    
    with open(analysis_file, 'w') as f:
        f.write("# Optimal Paper Results Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n")
        f.write(f"- **Total Tests**: {len(results)}\n")
        f.write(f"- **Successful Tests**: {len(successful_results)}\n")
        f.write(f"- **Success Rate**: {len(successful_results)/len(results)*100:.1f}%\n\n")
        
        if successful_results:
            avg_accuracy = sum(r['accuracy'] for r in successful_results) / len(successful_results)
            avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
            avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)
            
            f.write(f"- **Average Accuracy**: {avg_accuracy:.2f}%\n")
            f.write(f"- **Average Precision**: {avg_precision:.2f}%\n")
            f.write(f"- **Average Recall**: {avg_recall:.2f}%\n\n")
        
        f.write("## Detailed Results by Dataset\n\n")
        
        for dataset in ['MNIST', 'CIFAR10', 'Alzheimer']:
            dataset_results = [r for r in successful_results if r['dataset'] == dataset]
            if dataset_results:
                f.write(f"### {dataset}\n")
                f.write("| Attack Type | Accuracy | Precision | Recall | F1-Score | Quality |\n")
                f.write("|-------------|----------|-----------|--------|----------|----------|\n")
                
                for result in dataset_results:
                    f.write(f"| {result['attack_type']} | {result['accuracy']:.2f}% | "
                           f"{result['precision']:.2f}% | {result['recall']:.2f}% | "
                           f"{result['f1_score']:.2f}% | {result['quality_score']} |\n")
                
                avg_acc = sum(r['accuracy'] for r in dataset_results) / len(dataset_results)
                target_acc = config['dataset_configs'][dataset]['expected_accuracy']
                f.write(f"\n**Average Accuracy**: {avg_acc:.2f}% (Target: {target_acc}%)\n\n")
        
        f.write("## Paper Readiness Assessment\n")
        excellent_results = [r for r in successful_results if r['quality_score'] == 'EXCELLENT']
        f.write(f"- **Excellent Results**: {len(excellent_results)}/{len(results)}\n")
        f.write(f"- **Paper Ready**: {'YES' if len(excellent_results) >= len(results) * 0.7 else 'NEEDS_IMPROVEMENT'}\n\n")

if __name__ == "__main__":
    print("🎯 OPTIMAL PAPER RESULTS TEST")
    print("=" * 50)
    print("This test is designed to achieve the best possible")
    print("results for your research paper submission.")
    print("Estimated time: 12-18 hours for complete analysis")
    print("=" * 50)
    
    choice = input("\nProceed with optimal test? (y/N): ").strip().lower()
    
    if choice in ['y', 'yes']:
        try:
            results_file, analysis_file = run_optimal_comprehensive_test()
            print(f"\n✅ SUCCESS! Results ready for paper submission")
            print(f"📄 {results_file}")
            print(f"📊 {analysis_file}")
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
    else:
        print("Test cancelled") 