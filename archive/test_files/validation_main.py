"""
Validation Main - Test our predictions with real experiments
This runs quick experiments to validate our Non-IID predictions
"""

import torch
import sys
import os
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import everything we need
from federated_learning.config import config
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds

def validate_prediction(dataset, non_iid_type, expected_accuracy, expected_detection):
    """Validate a specific prediction"""
    
    print(f"\n🧪 VALIDATING: {dataset} {non_iid_type}")
    print(f"Expected Accuracy: {expected_accuracy:.2f}%")
    print(f"Expected Detection: {expected_detection:.2f}%")
    print("="*50)
    
    try:
        # Configure for this test
        config.DATASET = dataset
        config.ENABLE_NON_IID = True
        
        if non_iid_type == "Dirichlet":
            config.DIRICHLET_ALPHA = 0.1
            config.LABEL_SKEW_RATIO = None
        else:  # Label Skew
            config.DIRICHLET_ALPHA = None
            config.LABEL_SKEW_RATIO = 0.7
        
        # Quick settings for fast validation
        config.GLOBAL_EPOCHS = 2
        config.LOCAL_EPOCHS_ROOT = 3
        config.LOCAL_EPOCHS_CLIENT = 2
        config.BATCH_SIZE = 16
        config.VAE_EPOCHS = 3
        config.DUAL_ATTENTION_EPOCHS = 2
        config.SHAPLEY_SAMPLES = 5
        
        # Set random seed for reproducibility
        set_random_seeds(42)
        
        # Load data
        root_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        
        # Create server
        server = Server()
        server.set_datasets(root_loader, test_dataset)
        server._pretrain_global_model()
        
        initial_accuracy = server.evaluate_model()
        
        # Create client datasets
        root_client_dataset, client_datasets = create_client_datasets(
            train_dataset=root_dataset,
            num_clients=config.NUM_CLIENTS,
            iid=False,  # Non-IID
            alpha=config.DIRICHLET_ALPHA if config.DIRICHLET_ALPHA else None
        )
        
        # Create clients
        clients = []
        for i in range(config.NUM_CLIENTS):
            client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
            clients.append(client)
        
        server.add_clients(clients)
        
        # Train VAE quickly
        root_gradients = server._collect_root_gradients()
        server.vae = server.train_vae(root_gradients, vae_epochs=config.VAE_EPOCHS)
        
        # Run one attack test (Partial Scaling - our best performer)
        import numpy as np
        num_malicious = int(config.NUM_CLIENTS * config.FRACTION_MALICIOUS)
        malicious_indices = np.random.choice(config.NUM_CLIENTS, num_malicious, replace=False)
        
        for i, client in enumerate(clients):
            if i in malicious_indices:
                client.is_malicious = True
                client.set_attack_parameters(
                    attack_type='partial_scaling_attack',
                    scaling_factor=config.SCALING_FACTOR,
                    partial_percent=config.PARTIAL_SCALING_PERCENT
                )
            else:
                client.is_malicious = False
        
        # Run federated learning
        training_errors, round_metrics = server.train(num_rounds=config.GLOBAL_EPOCHS)
        
        # Get final results
        final_accuracy = server.evaluate_model()
        
        # Calculate detection metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        if round_metrics:
            for round_idx, round_data in round_metrics.items():
                if 'detection_results' in round_data and round_data['detection_results']:
                    det_results = round_data['detection_results']
                    total_tp += det_results.get('true_positives', 0)
                    total_fp += det_results.get('false_positives', 0)
                    total_fn += det_results.get('false_negatives', 0)
        
        detection_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        detection_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        # Convert to percentages
        actual_accuracy = final_accuracy * 100
        actual_detection = detection_precision * 100
        
        # Calculate differences
        accuracy_diff = abs(actual_accuracy - expected_accuracy)
        detection_diff = abs(actual_detection - expected_detection)
        
        result = {
            'dataset': dataset,
            'non_iid_type': non_iid_type,
            'expected_accuracy': expected_accuracy,
            'actual_accuracy': actual_accuracy,
            'accuracy_difference': accuracy_diff,
            'expected_detection': expected_detection,
            'actual_detection': actual_detection,
            'detection_difference': detection_diff,
            'validation_success': accuracy_diff <= 5.0 and detection_diff <= 10.0,
            'initial_accuracy': initial_accuracy * 100,
            'malicious_indices': malicious_indices.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ ACTUAL RESULTS:")
        print(f"   Accuracy: {actual_accuracy:.2f}% (diff: {accuracy_diff:.2f}pp)")
        print(f"   Detection: {actual_detection:.2f}% (diff: {detection_diff:.2f}pp)")
        
        if result['validation_success']:
            print(f"🎯 VALIDATION SUCCESS - Predictions accurate!")
        else:
            print(f"⚠️ PREDICTION NEEDS ADJUSTMENT")
        
        return result
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset,
            'non_iid_type': non_iid_type,
            'error': str(e),
            'validation_success': False,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Run validation tests for our key predictions"""
    
    print("🔬 COMPREHENSIVE PREDICTION VALIDATION")
    print("="*60)
    
    # Test our most confident predictions
    validation_tests = [
        # MNIST predictions (should be most accurate)
        ('MNIST', 'Label_Skew', 97.45, 51.8),
        ('MNIST', 'Dirichlet', 97.12, 33.1),
        
        # Alzheimer predictions (medical domain)
        ('Alzheimer', 'Label_Skew', 95.1, 58.5),
        ('Alzheimer', 'Dirichlet', 94.8, 45.0),
        
        # CIFAR-10 predictions (most challenging)
        ('CIFAR-10', 'Label_Skew', 79.8, 31.5),
        ('CIFAR-10', 'Dirichlet', 78.6, 28.0)
    ]
    
    validation_results = []
    
    for dataset, non_iid_type, expected_acc, expected_det in validation_tests:
        result = validate_prediction(dataset, non_iid_type, expected_acc, expected_det)
        validation_results.append(result)
        
        print("\n" + "="*50)
    
    # Summary analysis
    print(f"\n📊 VALIDATION SUMMARY")
    print("="*60)
    
    successful_validations = [r for r in validation_results if r.get('validation_success', False)]
    total_tests = len([r for r in validation_results if 'error' not in r])
    
    print(f"Successful validations: {len(successful_validations)}/{total_tests}")
    print(f"Success rate: {len(successful_validations)/total_tests*100:.1f}%")
    
    # Calculate average differences
    accuracy_diffs = [r.get('accuracy_difference', 100) for r in validation_results if 'accuracy_difference' in r]
    detection_diffs = [r.get('detection_difference', 100) for r in validation_results if 'detection_difference' in r]
    
    if accuracy_diffs:
        avg_acc_diff = sum(accuracy_diffs) / len(accuracy_diffs)
        print(f"Average accuracy difference: {avg_acc_diff:.2f}pp")
    
    if detection_diffs:
        avg_det_diff = sum(detection_diffs) / len(detection_diffs)
        print(f"Average detection difference: {avg_det_diff:.2f}pp")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"results/validation_results_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'validation_summary': {
                'total_tests': total_tests,
                'successful_validations': len(successful_validations),
                'success_rate': len(successful_validations)/total_tests*100 if total_tests > 0 else 0,
                'average_accuracy_difference': avg_acc_diff if accuracy_diffs else None,
                'average_detection_difference': avg_det_diff if detection_diffs else None
            },
            'individual_results': validation_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return validation_results

if __name__ == "__main__":
    results = main() 