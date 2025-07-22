#!/usr/bin/env python3
"""
Comprehensive Final Test - Run all experiments for paper submission
Tests all 3 datasets with all 5 attacks and saves authentic results
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
import torch
import numpy as np

# Add federated_learning to path
sys.path.append('federated_learning')

def run_single_experiment(dataset, model, attack_type, experiment_name):
    """Run a single experiment with given parameters"""
    print(f"\n{'='*80}")
    print(f"🔬 EXPERIMENT: {experiment_name}")
    print(f"Dataset: {dataset} | Model: {model} | Attack: {attack_type}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import and setup
        from federated_learning.config import config
        from federated_learning.training.server import Server
        
        # Configure for this experiment
        config.DATASET = dataset
        config.MODEL = model
        
        # Dataset-specific configurations
        if dataset == 'MNIST':
            config.INPUT_CHANNELS = 1
            config.NUM_CLASSES = 10
            config.IMG_SIZE = 28
        elif dataset == 'CIFAR10':
            config.INPUT_CHANNELS = 3
            config.NUM_CLASSES = 10
            config.IMG_SIZE = 32
        elif dataset == 'Alzheimer':
            config.INPUT_CHANNELS = 3
            config.NUM_CLASSES = 4
            config.IMG_SIZE = 224
        
        # Force device configuration
        if torch.cuda.is_available():
            config.device = torch.device('cuda')
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            config.device = torch.device('cpu')
            print("⚠️ Using CPU")
        
        # Load data
        from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
        from federated_learning.training.client import Client
        from federated_learning.utils.model_utils import set_random_seeds
        
        # Set random seed for reproducibility
        set_random_seeds(42)
        
        # Load datasets
        root_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        
        # Initialize server
        server = Server()
        server.set_datasets(root_loader, test_dataset)
        server._pretrain_global_model()
        
        # Create client datasets
        root_client_dataset, client_datasets = create_client_datasets(
            train_dataset=root_dataset,
            num_clients=config.NUM_CLIENTS,
            iid=True  # Using IID for baseline
        )
        
        # Create clients
        clients = []
        for i in range(config.NUM_CLIENTS):
            client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
            clients.append(client)
        
        # Set malicious clients and attack types
        import numpy as np
        malicious_indices = np.random.choice(config.NUM_CLIENTS, config.NUM_MALICIOUS, replace=False)
        
        for i, client in enumerate(clients):
            if i in malicious_indices:
                client.is_malicious = True
                client.set_attack_parameters(
                    attack_type=attack_type,
                    scaling_factor=config.SCALING_FACTOR if hasattr(config, 'SCALING_FACTOR') else 10.0,
                    partial_percent=config.PARTIAL_SCALING_PERCENT if hasattr(config, 'PARTIAL_SCALING_PERCENT') else 0.5
                )
            else:
                client.is_malicious = False
        
        # Add clients to server
        server.add_clients(clients)
        
        # Train VAE
        root_gradients = server._collect_root_gradients()
        server.vae = server.train_vae(root_gradients, vae_epochs=config.VAE_EPOCHS)
        
        print(f"🚀 Starting training with {config.NUM_CLIENTS} clients ({config.NUM_MALICIOUS} malicious)")
        print(f"📊 Attack type: {attack_type}")
        print(f"🔧 Malicious clients: {malicious_indices.tolist()}")
        
        # Run training
        training_errors, round_metrics = server.train(num_rounds=config.GLOBAL_EPOCHS)
        
        # Calculate metrics
        execution_time = time.time() - start_time
        
        # Extract final metrics
        final_accuracy = server.evaluate_model()
        
        # Calculate detection metrics from round metrics
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        
        if round_metrics:
            for round_idx, round_data in round_metrics.items():
                if 'detection_results' in round_data and round_data['detection_results']:
                    det_results = round_data['detection_results']
                    total_tp += det_results.get('true_positives', 0)
                    total_fp += det_results.get('false_positives', 0)
                    total_tn += det_results.get('true_negatives', 0)
                    total_fn += det_results.get('false_negatives', 0)
        
        # Use totals for detection metrics
        true_positives = total_tp
        false_positives = total_fp
        true_negatives = total_tn
        false_negatives = total_fn
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        experiment_result = {
            'experiment_name': experiment_name,
            'dataset': dataset,
            'model': model,
            'attack_type': attack_type,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'execution_time': execution_time,
            'final_accuracy': final_accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'num_clients': config.NUM_CLIENTS,
            'num_malicious': config.NUM_MALICIOUS,
            'global_epochs': config.GLOBAL_EPOCHS,
            'local_epochs': config.LOCAL_EPOCHS_CLIENT,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LR,
        }
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   ✅ Accuracy: {final_accuracy:.4f}")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   🔍 Recall: {recall:.4f}")
        print(f"   📈 F1-Score: {f1_score:.4f}")
        print(f"   ⏱️ Time: {execution_time:.2f}s")
        print(f"   🔢 Confusion Matrix: TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}")
        
        return experiment_result
        
    except Exception as e:
        print(f"❌ EXPERIMENT FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'experiment_name': experiment_name,
            'dataset': dataset,
            'model': model,
            'attack_type': attack_type,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'status': 'FAILED',
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def main():
    """Run comprehensive experiments for all datasets and attacks"""
    
    print("\n" + "="*100)
    print("🔬 COMPREHENSIVE FINAL VALIDATION TEST")
    print("📊 Testing all datasets with all attack types")
    print("="*100)
    
    # Define experiment configurations
    experiments = [
        # MNIST Experiments
        {'dataset': 'MNIST', 'model': 'CNN', 'attacks': [
            'scaling_attack', 'partial_scaling_attack', 'sign_flipping_attack', 
            'noise_attack', 'label_flipping'
        ]},
        
        # CIFAR-10 Experiments
        {'dataset': 'CIFAR10', 'model': 'ResNet18', 'attacks': [
            'scaling_attack', 'partial_scaling_attack', 'sign_flipping_attack', 
            'noise_attack', 'label_flipping'
        ]},
        
        # Alzheimer Experiments
        {'dataset': 'Alzheimer', 'model': 'ResNet18', 'attacks': [
            'scaling_attack', 'partial_scaling_attack', 'sign_flipping_attack', 
            'noise_attack', 'label_flipping'
        ]}
    ]
    
    all_results = []
    total_experiments = sum(len(exp['attacks']) for exp in experiments)
    current_experiment = 0
    
    print(f"📋 Total experiments planned: {total_experiments}")
    print(f"⏰ Estimated time: {total_experiments * 3:.0f}-{total_experiments * 8:.0f} minutes")
    
    for dataset_config in experiments:
        dataset = dataset_config['dataset']
        model = dataset_config['model']
        attacks = dataset_config['attacks']
        
        print(f"\n🎯 Starting {dataset} experiments with {model}")
        
        for attack_type in attacks:
            current_experiment += 1
            experiment_name = f"{dataset}_{model}_{attack_type}_{current_experiment:02d}"
            
            print(f"\n[{current_experiment}/{total_experiments}] Running: {experiment_name}")
            
            result = run_single_experiment(dataset, model, attack_type, experiment_name)
            all_results.append(result)
            
            # Small pause between experiments
            time.sleep(2)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory
    results_dir = f"results/final_paper_submission_ready"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(all_results)
    csv_path = f"{results_dir}/COMPREHENSIVE_FINAL_RESULTS_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = f"{results_dir}/COMPREHENSIVE_FINAL_RESULTS_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
    summary_path = f"{results_dir}/FINAL_EXPERIMENT_SUMMARY_{timestamp}.md"
    
    print(f"\n" + "="*100)
    print("📊 GENERATING FINAL SUMMARY REPORT")
    print("="*100)
    
    with open(summary_path, 'w') as f:
        f.write("# Final Comprehensive Experiment Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(all_results)}\n\n")
        
        # Summary by dataset
        for dataset_config in experiments:
            dataset = dataset_config['dataset']
            model = dataset_config['model']
            
            f.write(f"## {dataset} + {model} Results\n\n")
            
            dataset_results = [r for r in all_results if r['dataset'] == dataset and r.get('status') != 'FAILED']
            
            if dataset_results:
                f.write("| Attack Type | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN |\n")
                f.write("|-------------|----------|-----------|--------|----------|----|----|----|----|  \n")
                
                for result in dataset_results:
                    f.write(f"| {result['attack_type']} | {result.get('final_accuracy', 0):.4f} | "
                           f"{result.get('precision', 0):.4f} | {result.get('recall', 0):.4f} | "
                           f"{result.get('f1_score', 0):.4f} | {result.get('true_positives', 0)} | "
                           f"{result.get('false_positives', 0)} | {result.get('true_negatives', 0)} | "
                           f"{result.get('false_negatives', 0)} |\n")
                
                # Calculate averages
                avg_accuracy = np.mean([r.get('final_accuracy', 0) for r in dataset_results])
                avg_precision = np.mean([r.get('precision', 0) for r in dataset_results])
                avg_recall = np.mean([r.get('recall', 0) for r in dataset_results])
                avg_f1 = np.mean([r.get('f1_score', 0) for r in dataset_results])
                
                f.write(f"\n**Average Performance:**\n")
                f.write(f"- Accuracy: {avg_accuracy:.4f}\n")
                f.write(f"- Precision: {avg_precision:.4f}\n")
                f.write(f"- Recall: {avg_recall:.4f}\n")
                f.write(f"- F1-Score: {avg_f1:.4f}\n\n")
            else:
                f.write("No successful results for this dataset.\n\n")
    
    print(f"\n✅ COMPREHENSIVE EXPERIMENTS COMPLETED!")
    print(f"📊 Results saved to:")
    print(f"   📄 CSV: {csv_path}")
    print(f"   📋 JSON: {json_path}")
    print(f"   📖 Summary: {summary_path}")
    
    # Print quick summary
    successful_results = [r for r in all_results if r.get('status') != 'FAILED']
    failed_results = [r for r in all_results if r.get('status') == 'FAILED']
    
    print(f"\n📈 EXECUTION SUMMARY:")
    print(f"   ✅ Successful experiments: {len(successful_results)}/{len(all_results)}")
    print(f"   ❌ Failed experiments: {len(failed_results)}")
    
    if successful_results:
        avg_accuracy = np.mean([r.get('final_accuracy', 0) for r in successful_results])
        avg_precision = np.mean([r.get('precision', 0) for r in successful_results])
        print(f"   📊 Overall average accuracy: {avg_accuracy:.4f}")
        print(f"   🎯 Overall average precision: {avg_precision:.4f}")

if __name__ == "__main__":
    main() 