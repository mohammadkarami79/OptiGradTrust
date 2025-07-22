#!/usr/bin/env python3
"""
Single Dataset Test - Run focused experiments for MNIST dataset
Ensures proper execution and authentic results
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

def run_mnist_experiments():
    """Run all attack experiments for MNIST dataset"""
    
    print("\n" + "="*80)
    print("🔬 MNIST DATASET COMPREHENSIVE TEST")
    print("📊 Testing all 5 attack types with CNN model")
    print("="*80)
    
    # Import and setup
    from federated_learning.config import config
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds
    
    # Configure for MNIST
    config.DATASET = 'MNIST'
    config.MODEL = 'CNN' 
    config.INPUT_CHANNELS = 1
    config.NUM_CLASSES = 10
    config.ENABLE_NON_IID = False  # IID for baseline
    
    # Set random seed
    set_random_seeds(42)
    
    # Attack types to test
    attack_types = [
        'scaling_attack',
        'partial_scaling_attack', 
        'sign_flipping_attack',
        'noise_attack',
        'label_flipping'
    ]
    
    results = []
    
    for i, attack_type in enumerate(attack_types):
        print(f"\n[{i+1}/5] 🚀 Testing attack: {attack_type}")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Load datasets
            root_dataset, test_dataset = load_dataset()
            root_loader = torch.utils.data.DataLoader(
                root_dataset, 
                batch_size=config.BATCH_SIZE, 
                shuffle=True, 
                num_workers=0
            )
            
            # Initialize server
            server = Server()
            server.set_datasets(root_loader, test_dataset)
            print("✅ Server initialized")
            
            # Pretrain global model
            print("🔧 Pretraining global model...")
            server._pretrain_global_model()
            initial_accuracy = server.evaluate_model()
            print(f"✅ Initial accuracy: {initial_accuracy:.4f}")
            
            # Create client datasets
            root_client_dataset, client_datasets = create_client_datasets(
                train_dataset=root_dataset,
                num_clients=config.NUM_CLIENTS,
                iid=True  # Using IID for baseline
            )
            
            # Create clients
            clients = []
            for j in range(config.NUM_CLIENTS):
                client = Client(client_id=j, dataset=client_datasets[j], is_malicious=False)
                clients.append(client)
            print(f"✅ Created {len(clients)} clients")
            
            # Set malicious clients and attack types
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
            
            print(f"🎯 Malicious clients: {malicious_indices.tolist()}")
            
            # Add clients to server
            server.add_clients(clients)
            
            # Train VAE
            print("🔧 Training VAE...")
            root_gradients = server._collect_root_gradients()
            server.vae = server.train_vae(root_gradients, vae_epochs=config.VAE_EPOCHS)
            print("✅ VAE training completed")
            
            # Run federated training
            print(f"🚀 Starting federated training ({config.GLOBAL_EPOCHS} rounds)...")
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
                'dataset': 'MNIST',
                'model': 'CNN',
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
            
            print(f"\n📊 RESULTS:")
            print(f"   ✅ Final Accuracy: {final_accuracy:.4f}")
            print(f"   🎯 Precision: {precision:.4f}")
            print(f"   🔍 Recall: {recall:.4f}")
            print(f"   📈 F1-Score: {f1_score:.4f}")
            print(f"   ⏱️ Time: {execution_time:.1f}s")
            print(f"   🔢 TP={total_tp}, FP={total_fp}, TN={total_tn}, FN={total_fn}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ ATTACK {attack_type} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result = {
                'dataset': 'MNIST',
                'model': 'CNN',
                'attack_type': attack_type,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'status': 'FAILED',
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            results.append(result)
        
        # Brief pause between attacks
        print("⏸️ Brief pause...")
        time.sleep(3)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = "results/final_paper_submission_ready"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = f"{results_dir}/MNIST_COMPREHENSIVE_RESULTS_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = f"{results_dir}/MNIST_COMPREHENSIVE_RESULTS_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    successful_results = [r for r in results if r.get('status') == 'SUCCESS']
    
    print(f"\n" + "="*80)
    print("📊 MNIST EXPERIMENT SUMMARY")
    print("="*80)
    print(f"✅ Successful experiments: {len(successful_results)}/5")
    print(f"❌ Failed experiments: {5 - len(successful_results)}")
    
    if successful_results:
        avg_accuracy = np.mean([r['final_accuracy'] for r in successful_results])
        avg_precision = np.mean([r['precision'] for r in successful_results])
        avg_recall = np.mean([r['recall'] for r in successful_results])
        avg_f1 = np.mean([r['f1_score'] for r in successful_results])
        
        print(f"\n📈 AVERAGE PERFORMANCE:")
        print(f"   Accuracy: {avg_accuracy:.4f}")
        print(f"   Precision: {avg_precision:.4f}")
        print(f"   Recall: {avg_recall:.4f}")
        print(f"   F1-Score: {avg_f1:.4f}")
        
        print(f"\n📋 INDIVIDUAL RESULTS:")
        for result in successful_results:
            print(f"   {result['attack_type']:20s}: Acc={result['final_accuracy']:.4f}, Prec={result['precision']:.4f}")
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 CSV: {csv_path}")
    print(f"   📋 JSON: {json_path}")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting MNIST comprehensive test...")
    run_mnist_experiments()
    print("✅ MNIST test completed!") 