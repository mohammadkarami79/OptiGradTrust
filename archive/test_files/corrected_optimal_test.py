#!/usr/bin/env python3
"""
Corrected Optimal Test - Using proper config approach
Based on the working memory-optimized test, but with optimal parameters
"""

import os
import sys
import time
import gc
import torch
import pandas as pd
from datetime import datetime

# Add the federated learning module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'federated_learning'))

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def run_single_attack_test(dataset, model, attack_type, optimal_params=True):
    """Run single attack test with optimal or memory-optimized parameters"""
    print(f"\n{'='*60}")
    print(f"TEST: {dataset} + {model} + {attack_type}")
    print(f"Parameters: {'OPTIMAL' if optimal_params else 'MEMORY-OPTIMIZED'}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        clear_gpu_memory()
        
        # Import and configure
        import federated_learning.config.config as config
        from federated_learning.training.server import FederatedServer
        
        # Set dataset and model
        config.DATASET = dataset
        config.MODEL = model
        config.ATTACK_TYPE = attack_type
        
        # Set dataset-specific parameters
        if dataset == 'MNIST':
            config.INPUT_CHANNELS = 1
            config.NUM_CLASSES = 10
            if optimal_params:
                config.BATCH_SIZE = 64
                config.GLOBAL_EPOCHS = 20
                config.LOCAL_EPOCHS_CLIENT = 15
                config.LEARNING_RATE = 0.01
            print(f"🎯 Target: 99%+ accuracy")
        elif dataset == 'CIFAR10':
            config.INPUT_CHANNELS = 3
            config.NUM_CLASSES = 10
            if optimal_params:
                config.BATCH_SIZE = 32
                config.GLOBAL_EPOCHS = 25
                config.LOCAL_EPOCHS_CLIENT = 20
                config.LEARNING_RATE = 0.001
            print(f"🎯 Target: 85%+ accuracy")
        elif dataset == 'Alzheimer':
            config.INPUT_CHANNELS = 3
            config.NUM_CLASSES = 4
            if optimal_params:
                config.BATCH_SIZE = 16
                config.GLOBAL_EPOCHS = 25
                config.LOCAL_EPOCHS_CLIENT = 18
                config.LEARNING_RATE = 0.003
            print(f"🎯 Target: 96%+ accuracy")
        
        print(f"📊 Dataset: {dataset}")
        print(f"🧠 Model: {model}")
        print(f"⚠️  Attack: {attack_type}")
        print(f"🔄 Global Epochs: {config.GLOBAL_EPOCHS}")
        print(f"📈 Client Epochs: {config.LOCAL_EPOCHS_CLIENT}")
        print(f"📦 Batch Size: {config.BATCH_SIZE}")
        print(f"📚 Learning Rate: {config.LEARNING_RATE}")
        
        # Initialize and run
        server = FederatedServer()
        print("✅ Server initialized")
        
        print("🚀 Starting training...")
        final_model, metrics = server.train()
        
        print("📊 Evaluating...")
        accuracy = server.evaluate_model(final_model)
        detection_metrics = server.get_detection_metrics()
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        accuracy_pct = accuracy * 100
        precision = detection_metrics.get('precision', 0.0) * 100
        recall = detection_metrics.get('recall', 0.0) * 100
        f1_score = detection_metrics.get('f1_score', 0.0) * 100
        
        # Quality assessment
        target_acc = {'MNIST': 99.0, 'CIFAR10': 85.0, 'Alzheimer': 96.0}[dataset]
        quality = "EXCELLENT" if accuracy_pct >= target_acc else "GOOD" if accuracy_pct >= target_acc * 0.9 else "NEEDS_IMPROVEMENT"
        
        print(f"\n📈 RESULTS:")
        print(f"   Accuracy: {accuracy_pct:.2f}% (Target: {target_acc}%+)")
        print(f"   Precision: {precision:.2f}%")
        print(f"   Recall: {recall:.2f}%")
        print(f"   F1-Score: {f1_score:.2f}%")
        print(f"   Quality: {quality}")
        print(f"   Time: {execution_time/60:.1f} minutes")
        
        result = {
            'dataset': dataset,
            'model': model,
            'attack_type': attack_type,
            'accuracy': accuracy_pct,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'execution_time': execution_time,
            'quality': quality,
            'target_accuracy': target_acc,
            'parameters': 'OPTIMAL' if optimal_params else 'MEMORY_OPTIMIZED',
            'status': 'SUCCESS'
        }
        
        # Cleanup
        del server, final_model
        clear_gpu_memory()
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ ERROR: {str(e)}")
        
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
            'quality': 'FAILED',
            'target_accuracy': 0.0,
            'parameters': 'OPTIMAL' if optimal_params else 'MEMORY_OPTIMIZED',
            'status': f'ERROR: {str(e)}'
        }

if __name__ == "__main__":
    print("🎯 CORRECTED OPTIMAL TEST")
    print("=" * 50)
    print("Testing with corrected config approach")
    print("Choose: MNIST (fastest) or single test")
    print("=" * 50)
    
    dataset = input("Dataset (MNIST/CIFAR10/Alzheimer): ").strip()
    
    if dataset == 'MNIST':
        model = 'CNN'
    else:
        model = 'ResNet18'
    
    attack = input("Attack type (partial_scaling_attack recommended): ").strip() or 'partial_scaling_attack'
    
    print(f"\n🚀 Running {dataset} + {model} + {attack}")
    
    result = run_single_attack_test(dataset, model, attack, optimal_params=True)
    
    print(f"\n📊 FINAL RESULT:")
    print(f"   Dataset: {result['dataset']}")
    print(f"   Accuracy: {result['accuracy']:.2f}%")
    print(f"   Precision: {result['precision']:.2f}%")
    print(f"   Status: {result['status']}")
    print(f"   Quality: {result['quality']}")
    
    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results/final_paper_submission_ready"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = f"{results_dir}/CORRECTED_TEST_{timestamp}.csv"
    df = pd.DataFrame([result])
    df.to_csv(results_file, index=False)
    
    print(f"\n📄 Result saved: {results_file}") 