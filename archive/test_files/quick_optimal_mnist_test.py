#!/usr/bin/env python3
"""
Quick Optimal MNIST Test - Fixed version using working import pattern
Tests MNIST with all 5 attacks using optimal parameters for best results
"""

import os
import sys
import time
import gc
import torch
import pandas as pd
from datetime import datetime
import numpy as np

sys.path.append('federated_learning')

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def run_optimal_mnist_single_attack(attack_type):
    """Run single MNIST attack with optimal parameters"""
    print(f"\n{'='*60}")
    print(f"🎯 OPTIMAL MNIST TEST: {attack_type}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        clear_gpu_memory()
        
        # Import using working pattern
        from federated_learning.config import config
        from federated_learning.training.server import Server
        from federated_learning.training.client import Client
        from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
        from federated_learning.utils.model_utils import set_random_seeds
        
        # Set optimal MNIST configuration
        config.DATASET = 'MNIST'
        config.MODEL = 'CNN'
        config.INPUT_CHANNELS = 1
        config.NUM_CLASSES = 10
        
        # OPTIMAL parameters for best accuracy
        config.BATCH_SIZE = 64          # ✅ Increased from 16
        config.GLOBAL_EPOCHS = 15       # ✅ Increased from 5  
        config.LOCAL_EPOCHS_ROOT = 20   # ✅ Increased from 3
        config.LOCAL_EPOCHS_CLIENT = 8  # ✅ Increased from 2
        config.LEARNING_RATE = 0.01     # ✅ Optimal for MNIST
        config.VAE_EPOCHS = 15          # ✅ Increased from 3
        config.SHAPLEY_SAMPLES = 15     # ✅ Increased from 5
        
        # Memory management
        config.NUM_WORKERS = 0
        config.PIN_MEMORY = False
        config.VAE_DEVICE = 'cpu'
        
        set_random_seeds(42)
        
        print(f"📊 Dataset: MNIST")
        print(f"🧠 Model: CNN") 
        print(f"⚠️  Attack: {attack_type}")
        print(f"🔄 Global Epochs: {config.GLOBAL_EPOCHS}")
        print(f"📈 Root Epochs: {config.LOCAL_EPOCHS_ROOT}")
        print(f"📈 Client Epochs: {config.LOCAL_EPOCHS_CLIENT}")
        print(f"📦 Batch Size: {config.BATCH_SIZE}")
        print(f"🎯 Target: 95%+ accuracy")
        
        # Load dataset
        root_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(
            root_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0
        )
        clear_gpu_memory()
        
        # Initialize server
        server = Server()
        server.set_datasets(root_loader, test_dataset)
        clear_gpu_memory()
        
        # Pretrain with more epochs
        print("🚀 Pretraining global model...")
        server._pretrain_global_model()
        initial_accuracy = server.evaluate_model()
        clear_gpu_memory()
        
        print(f"📊 Initial accuracy: {initial_accuracy:.4f}")
        
        # Create clients
        root_client_dataset, client_datasets = create_client_datasets(
            train_dataset=root_dataset, 
            num_clients=config.NUM_CLIENTS, 
            iid=True
        )
        
        clients = []
        for i in range(config.NUM_CLIENTS):
            client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
            clients.append(client)
        
        # Set malicious clients with specified attack
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
        
        # Train VAE with more epochs
        print("🧠 Training VAE...")
        try:
            root_gradients = server._collect_root_gradients()
            if len(root_gradients) > 10:
                root_gradients = root_gradients[:10]
            server.vae = server.train_vae(root_gradients, vae_epochs=config.VAE_EPOCHS)
        except Exception as e:
            print(f"VAE training issue: {e}")
        
        clear_gpu_memory()
        
        # Run federated training
        print("🚀 Starting federated training...")
        training_errors, round_metrics = server.train(num_rounds=config.GLOBAL_EPOCHS)
        final_accuracy = server.evaluate_model()
        
        print(f"📊 Final accuracy: {final_accuracy:.4f}")
        
        # Calculate detection metrics
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
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        accuracy_pct = final_accuracy * 100
        precision_pct = precision * 100
        recall_pct = recall * 100
        f1_pct = f1_score * 100
        
        # Quality assessment
        quality = "EXCELLENT" if accuracy_pct >= 95.0 else "GOOD" if accuracy_pct >= 85.0 else "NEEDS_IMPROVEMENT"
        
        print(f"\n📈 RESULTS:")
        print(f"   Accuracy: {accuracy_pct:.2f}% (Target: 95%+)")
        print(f"   Precision: {precision_pct:.2f}%")
        print(f"   Recall: {recall_pct:.2f}%")
        print(f"   F1-Score: {f1_pct:.2f}%")
        print(f"   Quality: {quality}")
        print(f"   Time: {execution_time/60:.1f} minutes")
        
        result = {
            'attack_type': attack_type,
            'initial_accuracy': initial_accuracy * 100,
            'final_accuracy': accuracy_pct,
            'precision': precision_pct,
            'recall': recall_pct,
            'f1_score': f1_pct,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'true_negatives': total_tn,
            'false_negatives': total_fn,
            'execution_time': execution_time,
            'quality': quality,
            'status': 'SUCCESS'
        }
        
        # Cleanup
        del server, clients
        clear_gpu_memory()
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ ERROR: {str(e)}")
        
        clear_gpu_memory()
        
        return {
            'attack_type': attack_type,
            'initial_accuracy': 0.0,
            'final_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'execution_time': execution_time,
            'quality': 'FAILED',
            'status': f'ERROR: {str(e)}'
        }

def run_optimal_mnist_comprehensive():
    """Run MNIST with all 5 attacks using optimal parameters"""
    print("🎯 OPTIMAL MNIST COMPREHENSIVE TEST")
    print("=" * 60)
    print("🎯 OBJECTIVE: Achieve 95%+ accuracy with robust attack detection")
    print("📊 SCOPE: All 5 attacks with optimal parameters")
    print("⏱️  DURATION: ~2-3 hours")
    print("=" * 60)
    
    attack_types = [
        'partial_scaling_attack',  # Start with most effective
        'scaling_attack',
        'sign_flipping_attack',
        'noise_attack',
        'label_flipping_attack'
    ]
    
    results = []
    
    for i, attack_type in enumerate(attack_types, 1):
        print(f"\n⏳ Progress: {i}/{len(attack_types)}")
        
        result = run_optimal_mnist_single_attack(attack_type)
        results.append(result)
        
        # Progress summary
        successful = len([r for r in results if r['status'] == 'SUCCESS'])
        print(f"✅ Success: {successful}/{i}")
        
        # Brief pause between tests
        time.sleep(10)
    
    # Final analysis
    print(f"\n{'='*60}")
    print("📊 FINAL OPTIMAL MNIST RESULTS")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    
    if successful_results:
        avg_accuracy = sum(r['final_accuracy'] for r in successful_results) / len(successful_results)
        avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
        avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)
        excellent_count = len([r for r in successful_results if r['quality'] == 'EXCELLENT'])
        
        print(f"✅ Success Rate: {len(successful_results)}/{len(results)}")
        print(f"📊 Average Accuracy: {avg_accuracy:.2f}%")
        print(f"🎯 Average Precision: {avg_precision:.2f}%")
        print(f"🎯 Average Recall: {avg_recall:.2f}%")
        print(f"⭐ Excellent Results: {excellent_count}/{len(successful_results)}")
        
        # Paper readiness assessment
        paper_ready = avg_accuracy >= 90.0 and excellent_count >= len(successful_results) * 0.6
        print(f"📄 Paper Ready: {'YES ✅' if paper_ready else 'NEEDS_IMPROVEMENT ⚠️'}")
        
        if paper_ready:
            print("\n🎉 EXCELLENT! Results are optimal for paper submission")
            print("💡 These results significantly improve upon the 62.5% baseline")
        else:
            print("\n⚠️  Results show improvement but may need further optimization")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results/final_paper_submission_ready"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = f"{results_dir}/OPTIMAL_MNIST_COMPREHENSIVE_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    
    print(f"\n📄 Results saved: {results_file}")
    
    return results_file, successful_results

if __name__ == "__main__":
    print("🎯 OPTIMAL MNIST TEST - گزینه ب")
    print("=" * 50)
    print("هدف: دستیابی به دقت 95%+ برای MNIST")
    print("زمان تخمینی: 2-3 ساعت")
    print("=" * 50)
    
    choice = input("\nشروع تست بهینه MNIST؟ (y/N): ").strip().lower()
    
    if choice in ['y', 'yes', 'بله']:
        try:
            print("🚀 شروع تست جامع MNIST...")
            results_file, successful_results = run_optimal_mnist_comprehensive()
            
            if successful_results:
                avg_acc = sum(r['final_accuracy'] for r in successful_results) / len(successful_results)
                print(f"\n🎉 موفقیت! میانگین دقت: {avg_acc:.2f}%")
                
                if avg_acc >= 90.0:
                    print("✅ نتایج عالی برای مقاله!")
                else:
                    print("⚠️ نتایج بهبود یافته اما ممکن است نیاز به تنظیم بیشتری داشته باشد")
            else:
                print("❌ هیچ نتیجه موفقی بدست نیامد")
                
        except KeyboardInterrupt:
            print("\n⏹️  تست توسط کاربر متوقف شد")
        except Exception as e:
            print(f"\n❌ تست ناموفق: {e}")
    else:
        print("تست لغو شد") 