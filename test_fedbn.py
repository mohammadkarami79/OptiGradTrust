import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy

from federated_learning.training.aggregation import aggregate_gradients
from federated_learning.utils.model_utils import update_model_with_gradient

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SimpleCNNWithBN(nn.Module):
    """A simple CNN with BatchNorm layers for testing"""
    def __init__(self):
        super(SimpleCNNWithBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def print_bn_stats(model, prefix=""):
    """Print BatchNorm statistics for a model"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            print(f"{prefix}BatchNorm layer: {name}")
            print(f"{prefix}  Running mean: {module.running_mean.mean().item():.6f}")
            print(f"{prefix}  Running var: {module.running_var.mean().item():.6f}")
            if module.weight is not None:
                print(f"{prefix}  Weight: {module.weight.data.mean().item():.6f}")
            if module.bias is not None:
                print(f"{prefix}  Bias: {module.bias.data.mean().item():.6f}")

def model_to_vector(model):
    """Convert model parameters to a single vector"""
    vec = []
    for param in model.parameters():
        if param.requires_grad:
            vec.append(param.data.view(-1))
    return torch.cat(vec)

def train_model(model, data, targets, epochs=1):
    """Train a model on data and return the gradient"""
    # Create a copy of the initial model to compute gradients against
    initial_model = copy.deepcopy(model)
    
    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few steps
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Compute gradient (difference between current and initial parameters)
    grad_vector = model_to_vector(model) - model_to_vector(initial_model)
    
    return grad_vector, model

def create_synthetic_data():
    """Create synthetic data for testing"""
    # Create synthetic MNIST-like data
    batch_size = 10
    data = torch.randn(batch_size, 1, 28, 28)
    targets = torch.randint(0, 10, (batch_size,))
    return data, targets

def test_fedbn():
    """Test FedBN implementation"""
    print("\n=== Testing FedBN Implementation ===")
    
    # Create models and synthetic data for 3 clients
    num_clients = 3
    global_model = SimpleCNNWithBN()
    
    # Print initial BatchNorm stats
    print("Initial BatchNorm statistics:")
    print_bn_stats(global_model)
    
    # Collect client models and gradients
    client_models = []
    client_gradients = []
    
    print("\nTraining client models...")
    for i in range(num_clients):
        print(f"\nClient {i}:")
        # Create a copy of the global model for this client
        client_model = copy.deepcopy(global_model)
        
        # Create synthetic data for this client
        data, targets = create_synthetic_data()
        
        # Train the client model
        gradient, updated_client_model = train_model(client_model, data, targets, epochs=2)
        
        # Store the gradient and model
        client_gradients.append(gradient)
        client_models.append(updated_client_model)
        
        # Print BatchNorm stats after training
        print(f"Client {i} BatchNorm statistics after training:")
        print_bn_stats(updated_client_model, prefix="  ")
    
    # Test different aggregation methods
    test_methods = {
        'fedavg': False,  # FedAvg should NOT preserve BatchNorm
        'fedbn': True     # FedBN should preserve BatchNorm
    }
    
    for method_name, should_preserve in test_methods.items():
        print(f"\n--- Testing {method_name} ---")
        
        # Set up aggregation arguments
        kwargs = {'weights': torch.ones(num_clients) / num_clients}
        if method_name == 'fedbn':
            kwargs['model'] = global_model
            
        # Aggregate gradients
        print(f"Aggregating gradients with {method_name}...")
        aggregated_gradient = aggregate_gradients(
            client_gradients=client_gradients,
            aggregation_method=method_name,
            **kwargs
        )
        
        # Update global model with aggregated gradient
        preserve_bn = (method_name == 'fedbn')
        updated_global, total_change, avg_change = update_model_with_gradient(
            global_model, 
            aggregated_gradient, 
            learning_rate=0.01,
            proximal_mu=0.0,
            preserve_bn=preserve_bn
        )
        
        print(f"Model updated: total_change={total_change:.6f}, avg_change={avg_change:.6f}")
        
        # Print BatchNorm stats after aggregation
        print(f"BatchNorm statistics after {method_name}:")
        print_bn_stats(updated_global, prefix="  ")
        
        # Verify that BatchNorm parameters were preserved or aggregated as expected
        print(f"\nVerifying BatchNorm preservation for {method_name}:")
        
        all_preserved = True
        for name, module in updated_global.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Compare with the original model's BatchNorm parameters
                original_bn = None
                for orig_name, orig_module in global_model.named_modules():
                    if orig_name == name:
                        original_bn = orig_module
                        break
                
                if original_bn is not None:
                    # Check if running stats are equal (preserved)
                    mean_preserved = torch.allclose(module.running_mean, original_bn.running_mean)
                    var_preserved = torch.allclose(module.running_var, original_bn.running_var)
                    
                    if mean_preserved and var_preserved:
                        if should_preserve:
                            print(f"  ✓ {name}: Running stats preserved as expected")
                        else:
                            print(f"  ✗ {name}: Running stats preserved but should have been aggregated")
                            all_preserved = False
                    else:
                        if should_preserve:
                            print(f"  ✗ {name}: Running stats changed but should have been preserved")
                            all_preserved = False
                        else:
                            print(f"  ✓ {name}: Running stats aggregated as expected")
        
        # Print overall result
        if (should_preserve and all_preserved) or (not should_preserve and not all_preserved):
            print(f"\n{method_name} TEST PASSED: BatchNorm {'preservation' if should_preserve else 'aggregation'} working correctly")
        else:
            print(f"\n{method_name} TEST FAILED: BatchNorm {'preservation' if should_preserve else 'aggregation'} not working as expected")
    
    print("\n=== FedBN Test Completed ===")

if __name__ == "__main__":
    print("=== FedBN BatchNorm Preservation Test ===")
    test_fedbn() 