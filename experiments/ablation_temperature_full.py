"""
Full Ablation Study with Temperature Hybrid
===========================================

This comprehensive ablation study compares:
1. Temperature Hybrid (Full) - Our proposed method
2. Pure Dual Attention - Strong baseline
3. Without Shapley - Ablation test
4. Without VAE - Ablation test  
5. FedAvg baseline - Fair comparison

Duration: 6-8 hours
Rounds: 50 (sufficient for convergence)
Dataset: Alzheimer MRI
"""

import sys
import os
import json
import time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*80)
print("FULL ABLATION STUDY: Temperature Hybrid")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Import after path setup
import federated_learning.config.config as config
from federated_learning.training.server import Server
from federated_learning.data.dataset_utils import load_dataset
from federated_learning.utils.data_utils import create_client_datasets
from federated_learning.training.client import Client

# Base configuration
BASE_CONFIG = {
    'dataset': 'alzheimer',
    'model': 'ResNet18',
    'num_classes': 4,
    'input_channels': 3,
    'enable_non_iid': True,
    'dirichlet_alpha': 0.5,
    'global_epochs': 50,
    'num_clients': 10,
    'fraction_malicious': 0.3
}

# Experiment configurations
EXPERIMENTS = {
    'temperature_hybrid_full': {
        'name': 'Temperature Hybrid (Full)',
        'rl_method': 'temperature_hybrid',
        'enable_shapley': True,
        'enable_vae': True,
        'optimizer': 'fedbnp'
    },
    'pure_dual_attention': {
        'name': 'Pure Dual Attention',
        'rl_method': 'dual_attention',
        'enable_shapley': True,
        'enable_vae': True,
        'optimizer': 'fedbnp'
    },
    'without_shapley': {
        'name': 'Temperature Hybrid w/o Shapley',
        'rl_method': 'temperature_hybrid',
        'enable_shapley': False,
        'enable_vae': True,
        'optimizer': 'fedbnp'
    },
    'without_vae': {
        'name': 'Temperature Hybrid w/o VAE',
        'rl_method': 'temperature_hybrid',
        'enable_shapley': True,
        'enable_vae': False,
        'optimizer': 'fedbnp'
    },
    'fedavg_baseline': {
        'name': 'FedAvg Baseline',
        'rl_method': 'dual_attention',
        'enable_shapley': True,
        'enable_vae': True,
        'optimizer': 'fedavg'
    }
}

def run_experiment(exp_name, exp_config, base_config):
    """Run a single experiment configuration."""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_config['name']}")
    print("="*80)
    print(f"Configuration:")
    for key, value in exp_config.items():
        print(f"  {key}: {value}")
    
    # Set configuration
    config.DATASET = base_config['dataset']
    config.MODEL = base_config['model']
    config.NUM_CLASSES = base_config['num_classes']
    config.INPUT_CHANNELS = base_config['input_channels']
    config.ENABLE_NON_IID = base_config['enable_non_iid']
    config.DIRICHLET_ALPHA = base_config['dirichlet_alpha']
    config.GLOBAL_EPOCHS = base_config['global_epochs']
    config.NUM_CLIENTS = base_config['num_clients']
    config.FRACTION_MALICIOUS = base_config['fraction_malicious']
    
    config.RL_AGGREGATION_METHOD = exp_config['rl_method']
    config.ENABLE_SHAPLEY = exp_config['enable_shapley']
    config.ENABLE_VAE_TRAINING = exp_config['enable_vae']
    
    if exp_config['optimizer'] == 'fedavg':
        config.AGGREGATION_METHOD = 'fedavg'
    else:
        config.AGGREGATION_METHOD = 'fedbn'
        config.USE_FEDPROX = True
    
    # Enable necessary components
    config.ENABLE_DUAL_ATTENTION = True
    if 'temperature' in exp_config['rl_method'] or exp_config['rl_method'] == 'rl_actor_critic':
        config.ENABLE_RL = True
    else:
        config.ENABLE_RL = False
    
    try:
        # Load dataset
        print("\nLoading dataset...")
        root_dataset, test_dataset = load_dataset()
        print(f"  Training: {len(root_dataset)}, Test: {len(test_dataset)}")
        
        # Create server
        print("Creating server...")
        server = Server()
        server.set_datasets(root_dataset, test_dataset)
        
        # Pretrain
        print("Pre-training...")
        try:
            server._pretrain_global_model(epochs=2, batch_size=16)
        except Exception as e:
            print(f"  Pre-training skipped: {str(e)}")
        
        # Setup clients
        print("Setting up clients...")
        from federated_learning.utils.data_utils import split_data_among_clients
        client_datasets, _ = split_data_among_clients(
            root_dataset,
            num_clients=base_config['num_clients'],
            iid=not base_config['enable_non_iid'],
            dirichlet_alpha=base_config['dirichlet_alpha'] if base_config['enable_non_iid'] else None
        )
        
        clients = []
        for i, dataset in enumerate(client_datasets):
            client = Client(
                client_id=i,
                dataset=dataset,
                model_fn=server.model_fn,
                device=server.device
            )
            clients.append(client)
            server.add_clients([client])
        
        # Initial evaluation
        initial_acc = server.evaluate_model()
        print(f"\nInitial accuracy: {initial_acc:.4f}")
        
        # Train
        print(f"\nTraining for {base_config['global_epochs']} rounds...")
        start_time = time.time()
        server.train(num_rounds=base_config['global_epochs'])
        train_time = time.time() - start_time
        
        # Final evaluation
        final_acc = server.evaluate_model()
        improvement = final_acc - initial_acc
        
        print(f"\nFinal accuracy: {final_acc:.4f}")
        print(f"Improvement: {improvement:+.4f}")
        print(f"Training time: {train_time/60:.1f} minutes")
        
        # Return results
        return {
            'success': True,
            'initial_accuracy': float(initial_acc),
            'final_accuracy': float(final_acc),
            'improvement': float(improvement),
            'training_time_seconds': float(train_time),
            'configuration': exp_config,
            'error': None
        }
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'initial_accuracy': None,
            'final_accuracy': None,
            'improvement': None,
            'training_time_seconds': None,
            'configuration': exp_config,
            'error': str(e)
        }

# Main execution
if __name__ == '__main__':
    all_results = {}
    
    print(f"\nTotal experiments: {len(EXPERIMENTS)}")
    print(f"Estimated time: {len(EXPERIMENTS) * 1.5:.1f} hours\n")
    
    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items(), 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Starting: {exp_config['name']}")
        
        result = run_experiment(exp_name, exp_config, BASE_CONFIG)
        all_results[exp_name] = result
        
        if result['success']:
            print(f"[OK] {exp_config['name']}: {result['final_accuracy']:.4f}")
        else:
            print(f"[FAIL] {exp_config['name']}: {result['error']}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_dir = 'experiments/results/ablation_temperature'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/results_{timestamp}.json"
    
    final_output = {
        'timestamp': timestamp,
        'base_config': BASE_CONFIG,
        'experiments': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<35} {'Accuracy':<12} {'vs Full':<12} {'Status'}")
    print("-"*80)
    
    full_acc = all_results.get('temperature_hybrid_full', {}).get('final_accuracy', 0)
    
    for exp_name, result in all_results.items():
        config_name = EXPERIMENTS[exp_name]['name']
        if result['success']:
            acc = result['final_accuracy']
            diff = acc - full_acc if full_acc > 0 else 0
            status = "[OK]"
            print(f"{config_name:<35} {acc:>6.2%}      {diff:>+6.2%}      {status}")
        else:
            print(f"{config_name:<35} {'FAILED':<12} {'-':<12} [FAIL]")
    
    print("="*80)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext step:")
    print("  Analyze results and create LaTeX tables")
    print(f"  Results file: {output_file}")

