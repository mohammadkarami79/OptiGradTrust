#!/usr/bin/env python3
"""
CIFAR-10 Dataset Test - Run focused experiments
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

def run_cifar10_experiments():
    """Run all attack experiments for CIFAR-10 dataset"""
    
    print("🔬 CIFAR-10 DATASET COMPREHENSIVE TEST")
    
    # Import and setup
    from federated_learning.config import config
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds
    
    # Configure for CIFAR-10
    config.DATASET = 'CIFAR10'
    config.MODEL = 'ResNet18'
    config.INPUT_CHANNELS = 3
    config.NUM_CLASSES = 10
    config.ENABLE_NON_IID = False
    
    set_random_seeds(42)
    
    attack_types = ['scaling_attack', 'partial_scaling_attack', 'sign_flipping_attack', 'noise_attack', 'label_flipping']
    results = []
    
    for i, attack_type in enumerate(attack_types):
        print(f"[{i+1}/5] Testing attack: {attack_type}")
        start_time = time.time()
        
        try:
            # Load datasets
            root_dataset, test_dataset = load_dataset()
            root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
            
            # Initialize server
            server = Server()
            server.set_datasets(root_loader, test_dataset)
            server._pretrain_global_model()
            initial_accuracy = server.evaluate_model()
            
            # Create client datasets
            root_client_dataset, client_datasets = create_client_datasets(
                train_dataset=root_dataset,
                num_clients=config.NUM_CLIENTS,
                iid=True
            )
            
            # Create clients
            clients = []
            for j in range(config.NUM_CLIENTS):
                client = Client(client_id=j, dataset=client_datasets[j], is_malicious=False)
                clients.append(client)
            
            # Set malicious clients
            malicious_indices = np.random.choice(config.NUM_CLIENTS, config.NUM_MALICIOUS, replace=False)
            
            for j, client in enumerate(clients):
                if j in malicious_indices:
                    client.is_malicious = True
                    client.set_attack_parameters(
                        attack_type=attack_type,
                        scaling_factor=getattr(config, 'SCALING_FACTOR', 10.0),
                        partial_percent=getattr(config, 'PARTIAL_SCALING_PERCENT', 0.5)
                    )
                else:
                    client.is_malicious = False
            
            server.add_clients(clients)
            
            # Train VAE
            root_gradients = server._collect_root_gradients()
            server.vae = server.train_vae(root_gradients, vae_epochs=config.VAE_EPOCHS)
            
            # Run federated training
            training_errors, round_metrics = server.train(num_rounds=config.GLOBAL_EPOCHS)
            
            # Get final results
            final_accuracy = server.evaluate_model()
            execution_time = time.time() - start_time
            
            # Calculate detection metrics
            total_tp = total_fp = total_tn = total_fn = 0
            
            if round_metrics:
                for round_idx, round_data in round_metrics.items():
                    if 'detection_results' in round_data and round_data['detection_results']:
                        det_results = round_data['detection_results']
                        total_tp += det_results.get('true_positives', 0)
                        total_fp += det_results.get('false_positives', 0)
                        total_tn += det_results.get('true_negatives', 0)
                        total_fn += det_results.get('false_negatives', 0)
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            result = {
                'dataset': 'CIFAR10',
                'model': 'ResNet18',
                'attack_type': attack_type,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'execution_time': execution_time,
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'true_negatives': total_tn,
                'false_negatives': total_fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'malicious_indices': malicious_indices.tolist(),
                'num_clients': config.NUM_CLIENTS,
                'num_malicious': config.NUM_MALICIOUS,
                'status': 'SUCCESS'
            }
            
            print(f"RESULTS: Accuracy={final_accuracy:.4f}, Precision={precision:.4f}, Time={execution_time:.1f}s")
            results.append(result)
            
        except Exception as e:
            print(f"ATTACK {attack_type} FAILED: {str(e)}")
            result = {
                'dataset': 'CIFAR10',
                'model': 'ResNet18',
                'attack_type': attack_type,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'status': 'FAILED',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            results.append(result)
        
        time.sleep(3)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = "results/final_paper_submission_ready"
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = f"{results_dir}/CIFAR10_COMPREHENSIVE_RESULTS_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    json_path = f"{results_dir}/CIFAR10_COMPREHENSIVE_RESULTS_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    successful_results = [r for r in results if r.get('status') == 'SUCCESS']
    print(f"CIFAR-10 SUMMARY: {len(successful_results)}/5 successful experiments")
    
    if successful_results:
        avg_accuracy = np.mean([r['final_accuracy'] for r in successful_results])
        avg_precision = np.mean([r['precision'] for r in successful_results])
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
    
    print(f"Results saved to: {csv_path}")
    return results

if __name__ == "__main__":
    run_cifar10_experiments() 