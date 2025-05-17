import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

from federated_learning.training.aggregation import aggregate_gradients
from federated_learning.utils.model_utils import update_model_with_gradient, extract_bn_parameters, model_to_vector
from federated_learning.models.cnn import CNNMnist

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def create_synthetic_gradients(num_clients, dim, scale_factors=None):
    """Create synthetic gradients for testing"""
    if scale_factors is None:
        # Default: all gradients roughly similar
        scale_factors = [1.0] * num_clients
        
    gradients = []
    for i in range(num_clients):
        # Create base gradient with some randomness
        gradient = torch.randn(dim)
        
        # Scale according to factor
        gradient = gradient * scale_factors[i]
        
        # Normalize gradient
        if torch.norm(gradient) > 0:
            gradient = gradient / torch.norm(gradient)
            
        gradients.append(gradient)
        
    return gradients

def test_aggregation_methods():
    """Test all aggregation methods with synthetic gradients"""
    print("\n=== Testing All Aggregation Methods ===")
    
    # Create a model for FedBN tests
    model = CNNMnist()
    gradient_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Generate synthetic client gradients
    num_clients = 5
    
    # Create different test scenarios
    test_scenarios = [
        {
            'name': 'Uniform clients',
            'scale_factors': [1.0] * num_clients,
            'weights': torch.ones(num_clients) / num_clients
        },
        {
            'name': 'One scaled client (potential attack)',
            'scale_factors': [1.0, 1.0, 5.0, 1.0, 1.0],  # Client 2 has 5x scale
            'weights': torch.ones(num_clients) / num_clients
        },
        {
            'name': 'Weighted clients',
            'scale_factors': [1.0] * num_clients,
            'weights': torch.tensor([0.3, 0.2, 0.1, 0.2, 0.2])  # Different client weights
        }
    ]
    
    # Methods to test
    methods = [
        'fedavg',  # Standard federated averaging
        'fedprox',  # Federated Proximal
        'fedbn',  # Federated BatchNorm
        'fedbn_fedprox',  # Combined FedBN + FedProx
        'fednova',  # Federated Normalized Averaging
        'feddwa'  # Federated Dynamic Weight Averaging
    ]
    
    results = {}
    
    # Run tests for each scenario
    for scenario in test_scenarios:
        print(f"\n--- Test Scenario: {scenario['name']} ---")
        
        # Generate gradients for this scenario
        client_gradients = create_synthetic_gradients(
            num_clients=num_clients,
            dim=gradient_dim,
            scale_factors=scenario['scale_factors']
        )
        
        # Calculate gradient norms
        gradient_norms = [torch.norm(g).item() for g in client_gradients]
        print(f"Client gradient norms: {[f'{norm:.4f}' for norm in gradient_norms]}")
        
        # Test each aggregation method
        scenario_results = {}
        for method in methods:
            print(f"\nTesting {method}...")
            
            try:
                # Prepare kwargs based on method
                kwargs = {'weights': scenario['weights']}
                
                if method in ['fedbn', 'fedbn_fedprox']:
                    kwargs['model'] = model
                    
                if method in ['fedprox', 'fedbn_fedprox']:
                    kwargs['mu'] = 0.1
                
                # Run aggregation
                aggregated = aggregate_gradients(
                    client_gradients=client_gradients,
                    aggregation_method=method,
                    **kwargs
                )
                
                # Calculate aggregated norm
                agg_norm = torch.norm(aggregated).item()
                
                # Test model update
                model_copy = copy.deepcopy(model)
                updated_model, total_change, avg_change = update_model_with_gradient(
                    model_copy, 
                    aggregated, 
                    learning_rate=0.01,
                    proximal_mu=0.1 if 'prox' in method else 0.0,
                    preserve_bn='fedbn' in method
                )
                
                # For FedBN methods, check that BatchNorm parameters were preserved
                if 'fedbn' in method:
                    # Extract BatchNorm parameters before and after update
                    bn_params_before = extract_bn_parameters(model)
                    bn_params_after = extract_bn_parameters(updated_model)
                    
                    # Check if they are preserved
                    bn_preserved = all(
                        torch.allclose(bn_params_before[name], bn_params_after[name])
                        for name in bn_params_before
                    )
                    
                    if bn_preserved:
                        print(f"✅ {method}: BatchNorm parameters preserved correctly")
                    else:
                        print(f"❌ {method}: BatchNorm parameters not preserved")
                
                # Store results
                scenario_results[method] = {
                    'norm': agg_norm,
                    'total_change': total_change,
                    'avg_change': avg_change,
                    'bn_preserved': bn_preserved if 'fedbn' in method else 'N/A'
                }
                
                print(f"  Aggregated norm: {agg_norm:.4f}")
                print(f"  Model total change: {total_change:.4f}")
                print(f"  Model average change: {avg_change:.6f}")
                
            except Exception as e:
                print(f"❌ Error testing {method}: {str(e)}")
                import traceback
                traceback.print_exc()
                scenario_results[method] = {'error': str(e)}
        
        # Store scenario results
        results[scenario['name']] = scenario_results
    
    # Plot comparison of all methods across scenarios
    plot_comparison(results)
    
    return results

def plot_comparison(results):
    """Plot comparison of aggregation methods across scenarios"""
    # Set up the plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Prepare data for plotting
    scenarios = list(results.keys())
    methods = list(results[scenarios[0]].keys())
    
    # Plot gradient norms
    ax = axs[0]
    x = np.arange(len(scenarios))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        norms = []
        for scenario in scenarios:
            if 'norm' in results[scenario][method]:
                norms.append(results[scenario][method]['norm'])
            else:
                norms.append(0)
        
        ax.bar(x + i*width - 0.4, norms, width, label=method)
    
    ax.set_title('Aggregated Gradient Norm by Method and Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45)
    ax.set_ylabel('Gradient Norm')
    ax.legend()
    
    # Plot model change
    ax = axs[1]
    for i, method in enumerate(methods):
        changes = []
        for scenario in scenarios:
            if 'total_change' in results[scenario][method]:
                changes.append(results[scenario][method]['total_change'])
            else:
                changes.append(0)
        
        ax.bar(x + i*width - 0.4, changes, width, label=method)
    
    ax.set_title('Model Total Change by Method and Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45)
    ax.set_ylabel('Total Change')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('aggregation_comparison.png')
    plt.close()

def test_fedbn_parameter_preservation():
    """Specific test for FedBN's BatchNorm parameter preservation"""
    print("\n=== Testing FedBN BatchNorm Parameter Preservation ===")
    
    # Create a model with BatchNorm layers
    model = CNNMnist()
    
    # Identify all BatchNorm layers
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append((name, module))
    
    print(f"Found {len(bn_layers)} BatchNorm layers in the model")
    for name, _ in bn_layers:
        print(f"  {name}")
    
    # Create a gradient that would modify all parameters
    gradient = torch.randn(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Extract original BatchNorm parameters
    original_bn_params = extract_bn_parameters(model)
    
    # Update with FedAvg (should update BatchNorm)
    fedavg_model = copy.deepcopy(model)
    _, _, _ = update_model_with_gradient(
        fedavg_model, 
        gradient, 
        learning_rate=0.01,
        proximal_mu=0.0,
        preserve_bn=False
    )
    fedavg_bn_params = extract_bn_parameters(fedavg_model)
    
    # Update with FedBN (should preserve BatchNorm)
    fedbn_model = copy.deepcopy(model)
    _, _, _ = update_model_with_gradient(
        fedbn_model, 
        gradient, 
        learning_rate=0.01,
        proximal_mu=0.0,
        preserve_bn=True
    )
    fedbn_bn_params = extract_bn_parameters(fedbn_model)
    
    # Verify that BatchNorm parameters have changed with FedAvg
    fedavg_changed = False
    for name in original_bn_params:
        if not torch.allclose(original_bn_params[name], fedavg_bn_params[name]):
            fedavg_changed = True
            break
    
    # Verify that BatchNorm parameters are preserved with FedBN
    fedbn_preserved = True
    for name in original_bn_params:
        if not torch.allclose(original_bn_params[name], fedbn_bn_params[name]):
            fedbn_preserved = False
            break
    
    if fedavg_changed:
        print("✅ FedAvg correctly updates BatchNorm parameters")
    else:
        print("❌ FedAvg doesn't update BatchNorm parameters")
    
    if fedbn_preserved:
        print("✅ FedBN correctly preserves BatchNorm parameters")
    else:
        print("❌ FedBN doesn't preserve BatchNorm parameters")
    
    # Print detailed parameter changes for each BatchNorm layer
    print("\nDetailed BatchNorm parameter changes:")
    for name in original_bn_params:
        print(f"\n{name}:")
        orig = original_bn_params[name]
        avg = fedavg_bn_params[name]
        bn = fedbn_bn_params[name]
        
        # Calculate mean differences
        avg_diff = torch.mean(torch.abs(avg - orig)).item()
        bn_diff = torch.mean(torch.abs(bn - orig)).item()
        
        print(f"  Original mean: {torch.mean(orig).item():.6f}")
        print(f"  FedAvg mean: {torch.mean(avg).item():.6f}, diff: {avg_diff:.6f}")
        print(f"  FedBN mean: {torch.mean(bn).item():.6f}, diff: {bn_diff:.6f}")
    
    return fedavg_changed and fedbn_preserved

if __name__ == "__main__":
    print("=== Aggregation Methods Verification Tests ===")
    
    # Test all aggregation methods
    results = test_aggregation_methods()
    
    # Test FedBN parameter preservation specifically
    bn_test_pass = test_fedbn_parameter_preservation()
    
    # Overall test result
    if bn_test_pass:
        print("\n✅ FedBN parameter preservation working correctly")
    else:
        print("\n❌ FedBN parameter preservation has issues")
    
    print("\nTESTS COMPLETED") 