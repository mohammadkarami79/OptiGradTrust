"""
Medium Test for Temperature Hybrid - Alzheimer Dataset
=======================================================

This test verifies that:
1. Temperature hybrid works on Alzheimer dataset
2. Accuracy is in reasonable range (not 99% like MNIST)
3. Performance is acceptable (>85%)
4. System is stable over 10 rounds

Duration: ~30-40 minutes
Rounds: 10
Dataset: Alzheimer MRI
Clients: 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force settings
import federated_learning.config.config as config
config.DATASET = 'alzheimer'
config.MODEL = 'ResNet18'
config.NUM_CLASSES = 4
config.INPUT_CHANNELS = 3
config.ENABLE_NON_IID = True
config.DIRICHLET_ALPHA = 0.5
config.GLOBAL_EPOCHS = 10  # Medium test
config.NUM_CLIENTS = 10
config.FRACTION_MALICIOUS = 0.3
config.RL_AGGREGATION_METHOD = 'temperature_hybrid'
config.ENABLE_DUAL_ATTENTION = True
config.ENABLE_RL = True

print("="*80)
print("MEDIUM TEST: Temperature Hybrid on Alzheimer Dataset")
print("="*80)
print(f"Dataset: {config.DATASET}")
print(f"Model: {config.MODEL}")
print(f"Mode: {config.RL_AGGREGATION_METHOD}")
print(f"Rounds: {config.GLOBAL_EPOCHS}")
print(f"Clients: {config.NUM_CLIENTS}")
print(f"Malicious: {config.FRACTION_MALICIOUS * 100:.0f}%")
print(f"Non-IID: {config.ENABLE_NON_IID} (α={config.DIRICHLET_ALPHA})")
print("="*80)

from federated_learning.training.server import Server
from federated_learning.data.dataset_utils import load_dataset

# Load dataset
print("\n[1/5] Loading Alzheimer dataset...")
root_dataset, test_dataset = load_dataset()
print(f"  Training samples: {len(root_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# Create server
print("\n[2/5] Creating server...")
server = Server()
server.set_datasets(root_dataset, test_dataset)

# Pretrain
print("\n[3/5] Pre-training global model...")
try:
    server._pretrain_global_model(epochs=2, batch_size=16)
    print("  Pre-training completed")
except Exception as e:
    print(f"  Pre-training failed: {str(e)}")
    print("  Continuing without pre-training...")

# Setup clients
print("\n[4/5] Setting up clients...")
from federated_learning.utils.data_utils import create_federated_dataset
client_datasets, _, _ = create_federated_dataset(
    root_dataset,
    test_dataset,
    num_clients=config.NUM_CLIENTS,
    root_dataset_ratio=0.1,
    iid=not config.ENABLE_NON_IID,
    dirichlet_alpha=config.DIRICHLET_ALPHA if config.ENABLE_NON_IID else 1.0
)

from federated_learning.training.client import Client
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

print(f"  Created {len(clients)} clients")

# Show temperature schedule
print("\n" + "="*80)
print("TEMPERATURE SCHEDULE (10 rounds)")
print("="*80)
for round_idx in range(config.GLOBAL_EPOCHS):
    da_w, rl_w = server._compute_temperature_weights(round_idx, config.GLOBAL_EPOCHS)
    print(f"Round {round_idx:2d}: DA={da_w:.3f} ({da_w*100:5.1f}%), RL={rl_w:.3f} ({rl_w*100:5.1f}%)")

# Run training
print("\n" + "="*80)
print("[5/5] TRAINING WITH TEMPERATURE HYBRID")
print("="*80)

try:
    initial_acc = server.evaluate_model()
    print(f"Initial accuracy: {initial_acc:.4f}\n")
    
    # Train
    server.train(num_rounds=config.GLOBAL_EPOCHS)
    
    final_acc = server.evaluate_model()
    print(f"\nFinal accuracy: {final_acc:.4f}")
    print(f"Change: {final_acc - initial_acc:+.4f}")
    
    # Verification
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    checks = []
    
    # Check 1: Accuracy in reasonable range
    if 0.75 <= final_acc <= 0.99:
        checks.append(("Accuracy in reasonable range (75-99%)", True, final_acc))
    else:
        checks.append(("Accuracy in reasonable range", False, final_acc))
    
    # Check 2: Not MNIST accuracy (not 99%)
    if final_acc < 0.98:
        checks.append(("Not MNIST accuracy (< 98%)", True, final_acc))
    else:
        checks.append(("Not MNIST accuracy", False, final_acc))
    
    # Check 3: Acceptable performance (>85%)
    if final_acc >= 0.85:
        checks.append(("Acceptable performance (>= 85%)", True, final_acc))
    else:
        checks.append(("Acceptable performance", False, final_acc))
    
    # Print results
    passed = 0
    for check_name, result, value in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {check_name}: {value:.2%}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(checks)}")
    
    if passed == len(checks):
        print("\n" + "="*80)
        print("SUCCESS! Ready for Full Ablation Study")
        print("="*80)
        print("\nNext step:")
        print("  Run full ablation study (overnight)")
        print("  Command: python experiments/ablation_study_v2.py")
    else:
        print("\n" + "="*80)
        print("WARNING: Some checks failed")
        print("="*80)
        print("\nPlease review the results before proceeding.")
        print("Expected accuracy: 85-97% for Alzheimer dataset")
    
    # Save results
    import json
    results = {
        'test_type': 'medium_temperature_hybrid',
        'dataset': config.DATASET,
        'rounds': config.GLOBAL_EPOCHS,
        'clients': config.NUM_CLIENTS,
        'malicious_fraction': config.FRACTION_MALICIOUS,
        'initial_accuracy': float(initial_acc),
        'final_accuracy': float(final_acc),
        'improvement': float(final_acc - initial_acc),
        'checks_passed': passed,
        'checks_total': len(checks)
    }
    
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/temperature_medium_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: experiments/results/temperature_medium_test.json")
    
except Exception as e:
    print("\n" + "="*80)
    print("ERROR DURING TRAINING")
    print("="*80)
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\nPlease fix this before proceeding to full ablation study!")

