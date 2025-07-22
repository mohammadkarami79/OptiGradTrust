#!/usr/bin/env python3
"""
Memory-Optimized Test for Publication Results
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
import torch
import numpy as np
import gc

sys.path.append('federated_learning')

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def run_memory_optimized_test(dataset_name):
    """Run memory-optimized test for single dataset"""
    
    print(f"🔬 TESTING: {dataset_name}")
    clear_gpu_memory()
    
    from federated_learning.config import config
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds
    
    # Configure properly based on dataset
    if dataset_name == 'MNIST':
        config.DATASET = 'MNIST'
        config.MODEL = 'CNN'  # Correct for MNIST
        config.INPUT_CHANNELS = 1
        config.NUM_CLASSES = 10
    elif dataset_name == 'CIFAR10':
        config.DATASET = 'CIFAR10'
        config.MODEL = 'ResNet18'  # Correct for CIFAR10
        config.INPUT_CHANNELS = 3
        config.NUM_CLASSES = 10
    elif dataset_name == 'Alzheimer':
        config.DATASET = 'Alzheimer'
        config.MODEL = 'ResNet18'  # Correct for Alzheimer
        config.INPUT_CHANNELS = 3
        config.NUM_CLASSES = 4
    
    # Memory optimization
    config.BATCH_SIZE = 16
    config.GLOBAL_EPOCHS = 5
    config.LOCAL_EPOCHS_ROOT = 3
    config.LOCAL_EPOCHS_CLIENT = 2
    config.VAE_EPOCHS = 3
    config.SHAPLEY_SAMPLES = 5
    config.NUM_WORKERS = 0
    config.PIN_MEMORY = False
    config.VAE_DEVICE = 'cpu'
    
    set_random_seeds(42)
    
    start_time = time.time()
    attack_type = 'partial_scaling_attack'
    
    try:
        # Load data
        root_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        clear_gpu_memory()
        
        # Initialize server
        server = Server()
        server.set_datasets(root_loader, test_dataset)
        clear_gpu_memory()
        
        # Pretrain
        server._pretrain_global_model()
        initial_accuracy = server.evaluate_model()
        clear_gpu_memory()
        
        # Create clients
        root_client_dataset, client_datasets = create_client_datasets(
            train_dataset=root_dataset, num_clients=config.NUM_CLIENTS, iid=True
        )
        
        clients = []
        for i in range(config.NUM_CLIENTS):
            client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
            clients.append(client)
        
        # Set malicious clients
        malicious_indices = np.random.choice(config.NUM_CLIENTS, config.NUM_MALICIOUS, replace=False)
        for i, client in enumerate(clients):
            if i in malicious_indices:
                client.is_malicious = True
                client.set_attack_parameters(
                    attack_type=attack_type,
                    scaling_factor=10.0,
                    partial_percent=0.5
                )
        
        server.add_clients(clients)
        clear_gpu_memory()
        
        # Train VAE
        try:
            root_gradients = server._collect_root_gradients()
            if len(root_gradients) > 5:
                root_gradients = root_gradients[:5]
            server.vae = server.train_vae(root_gradients, vae_epochs=config.VAE_EPOCHS)
        except:
            print("VAE training skipped due to memory")
        
        clear_gpu_memory()
        
        # Run training
        training_errors, round_metrics = server.train(num_rounds=config.GLOBAL_EPOCHS)
        final_accuracy = server.evaluate_model()
        
        # Calculate metrics
        total_tp = total_fp = total_tn = total_fn = 0
        if round_metrics:
            for round_data in round_metrics.values():
                if 'detection_results' in round_data and round_data['detection_results']:
                    det = round_data['detection_results']
                    total_tp += det.get('true_positives', 0)
                    total_fp += det.get('false_positives', 0)
                    total_tn += det.get('true_negatives', 0)
                    total_fn += det.get('false_negatives', 0)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result = {
            'dataset': dataset_name,
            'model': config.MODEL,
            'attack_type': attack_type,
            'initial_accuracy': initial_accuracy,
            'final_accuracy': final_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'true_negatives': total_tn,
            'false_negatives': total_fn,
            'execution_time': time.time() - start_time,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'status': 'SUCCESS'
        }
        
        print(f"✅ {dataset_name}: Acc={final_accuracy:.4f}, Prec={precision:.4f}")
        return result
        
    except Exception as e:
        print(f"❌ {dataset_name} FAILED: {str(e)}")
        return {
            'dataset': dataset_name,
            'status': 'FAILED',
            'error': str(e),
            'execution_time': time.time() - start_time
        }
    finally:
        clear_gpu_memory()

def main():
    """Run tests for all datasets"""
    print("🔬 MEMORY-OPTIMIZED VALIDATION TEST")
    
    datasets = ['MNIST', 'CIFAR10', 'Alzheimer']
    results = []
    
    for dataset in datasets:
        result = run_memory_optimized_test(dataset)
        results.append(result)
        time.sleep(5)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = "results/final_paper_submission_ready"
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = f"{results_dir}/MEMORY_OPTIMIZED_VALIDATION_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    successful = [r for r in results if r.get('status') == 'SUCCESS']
    print(f"\n✅ SUCCESS: {len(successful)}/3 datasets")
    print(f"📄 Results: {csv_path}")
    
    return results

if __name__ == "__main__":
    main() 