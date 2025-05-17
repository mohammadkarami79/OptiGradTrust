import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from federated_learning.training.training_utils import client_update
from federated_learning.config.config import FEDPROX_MU

class SimpleNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.log_softmax(x, dim=1)

def calculate_parameter_differences(initial_state, updated_state):
    """Calculate the differences between two model states."""
    max_diff = 0
    avg_diff = 0
    num_params = 0
    total_params = 0
    
    for key in initial_state:
        if 'running' not in key:  # Exclude running stats of batch norm
            # Convert tensors to float for calculation
            param1 = initial_state[key].float()
            param2 = updated_state[key].float()
            
            # Calculate differences
            param_diff = torch.abs(param2 - param1)
            max_diff = max(max_diff, param_diff.max().item())
            
            # Calculate average difference for this parameter
            param_avg = param_diff.mean().item()
            param_size = param_diff.numel()
            
            avg_diff += param_avg * param_size
            total_params += param_size
            num_params += 1
    
    # Calculate weighted average difference across all parameters
    avg_diff = avg_diff / total_params if total_params > 0 else 0
    
    return max_diff, avg_diff, num_params, total_params

def test_client_update():
    print("\n=== Testing Client Update ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters
    input_dim = 784  # e.g., MNIST
    num_classes = 10
    batch_size = 32
    num_samples = 100
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create synthetic data
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Test 1: Basic Update
    print("\nTest 1: Basic Client Update")
    try:
        # Initialize model and optimizer
        client_model = SimpleNet(input_dim=input_dim, num_classes=num_classes).to(device)
        optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
        
        # Get initial loss
        initial_loss = 0.0
        num_batches = 0
        client_model.eval()
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = client_model(data)
                loss = nn.NLLLoss()(output, target)
                initial_loss += loss.item()
                num_batches += 1
        initial_loss /= num_batches
        
        # Train model
        initial_state = {k: v.clone() for k, v in client_model.state_dict().items()}
        updated_state = client_update(client_model, optimizer, train_loader, epochs)
        
        # Get final loss
        final_loss = 0.0
        num_batches = 0
        client_model.eval()
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = client_model(data)
                loss = nn.NLLLoss()(output, target)
                final_loss += loss.item()
                num_batches += 1
        final_loss /= num_batches
        
        # Verify improvements
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        assert final_loss < initial_loss, "Training should reduce the loss"
        
        # Calculate parameter differences
        max_diff, avg_diff, num_params, total_params = calculate_parameter_differences(initial_state, updated_state)
        print(f"Parameter changes - Max: {max_diff:.6f}, Avg: {avg_diff:.6f}")
        assert max_diff > 0, "Model parameters should change after training"
        print("✓ Basic client update successful - parameters were updated and loss decreased")
    except Exception as e:
        print(f"✗ Basic client update failed: {str(e)}")
        raise
    
    # Test 2: FedProx Update
    print("\nTest 2: FedProx Client Update")
    try:
        # Initialize models
        client_model = SimpleNet(input_dim=input_dim, num_classes=num_classes).to(device)
        global_model = SimpleNet(input_dim=input_dim, num_classes=num_classes).to(device)
        optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
        
        # Train with FedProx
        initial_state = global_model.state_dict()
        client_model.load_state_dict(initial_state)
        updated_state = client_update(client_model, optimizer, train_loader, epochs, global_model)
        
        # Calculate parameter differences
        max_diff, avg_diff, num_params, total_params = calculate_parameter_differences(initial_state, updated_state)
        
        print(f"Maximum parameter difference: {max_diff:.6f}")
        print(f"Average parameter difference: {avg_diff:.6f}")
        print(f"Number of parameter tensors: {num_params}")
        print(f"Total number of parameters: {total_params}")
        
        # More realistic thresholds based on FedProx mu
        max_allowed_diff = 5.0 / FEDPROX_MU  # Inverse relationship with mu
        avg_allowed_diff = 1.0 / FEDPROX_MU
        
        assert max_diff > 0, "Model parameters should change"
        assert max_diff < max_allowed_diff, f"Max parameter difference ({max_diff:.6f}) exceeds threshold ({max_allowed_diff:.6f})"
        assert avg_diff < avg_allowed_diff, f"Average parameter difference ({avg_diff:.6f}) exceeds threshold ({avg_allowed_diff:.6f})"
        print("✓ FedProx client update successful - parameters were updated within constraints")
    except Exception as e:
        print(f"✗ FedProx client update failed: {str(e)}")
        raise
    
    # Test 3: Memory Usage
    print("\nTest 3: Testing Memory Usage")
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Train with larger dataset
        X_large = torch.randn(1000, input_dim)
        y_large = torch.randint(0, num_classes, (1000,))
        large_dataset = TensorDataset(X_large, y_large)
        large_loader = DataLoader(large_dataset, batch_size=batch_size, shuffle=True)
        
        _ = client_update(client_model, optimizer, large_loader, epochs=1)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = final_memory - initial_memory
        
        print(f"Memory usage difference: {memory_diff:.2f} MB")
        assert memory_diff < 1000, "Memory usage increase should be reasonable"
        print("✓ Memory usage test successful")
    except ImportError:
        print("⚠ Skipping memory test - psutil not available")
    except Exception as e:
        print(f"✗ Memory usage test failed: {str(e)}")
        raise
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_client_update() 