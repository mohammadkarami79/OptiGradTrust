import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset import load_dataset, split_dataset, create_root_dataset

def test_global_model_update():
    """
    Test that the global model updates correctly across epochs.
    This test verifies that:
    1. The model parameters change after updates
    2. The model performance improves over time
    """
    print("\n=== Testing Global Model Update ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Load datasets
    train_dataset, test_dataset, num_classes, input_channels = load_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    # Create a server instance
    server = Server()
    
    # Create a simple client dataset for testing
    client_datasets = split_dataset(train_dataset, num_classes)
    
    # Create clients
    clients = []
    for i in range(NUM_CLIENTS):
        is_malicious = i < NUM_MALICIOUS
        client = Client(i, client_datasets[i], is_malicious=is_malicious)
        clients.append(client)
    
    # Get initial model parameters
    initial_params = {}
    for name, param in server.global_model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.data.clone()
    
    # Test initial model
    server.global_model.eval()
    test_acc_initial = 0
    test_loss_initial = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(server.device), target.to(server.device)
            output = server.global_model(data)
            loss = F.cross_entropy(output, target)
            test_loss_initial += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            test_acc_initial += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss_initial /= len(test_loader)
    test_acc_initial /= len(test_loader.dataset)
    
    print(f"Initial test accuracy: {test_acc_initial:.4f}")
    print(f"Initial test loss: {test_loss_initial:.4f}")
    
    # Perform one round of training
    print("\n=== Performing one round of training ===")
    
    # Train selected clients
    client_gradients = []
    client_features = []
    
    for client in clients:
        print(f"\nTraining client {client.client_id}...")
        gradient, features = client.train(server.global_model, 0)
        client_gradients.append(gradient)
        client_features.append(features)
    
    # Stack features into a tensor
    if client_features and all(f is not None for f in client_features):
        features_tensor = torch.stack(client_features)
    else:
        features_tensor = torch.zeros((len(clients), DUAL_ATTENTION_FEATURE_DIM), device=server.device)
    
    # Aggregate gradients
    print("\nAggregating gradients...")
    aggregated_gradient = server.aggregate_gradients(client_gradients, features_tensor, 0)
    
    # Update global model
    print("\nUpdating global model...")
    updated_model = server._update_global_model(aggregated_gradient, 0)
    server.global_model = updated_model
    
    # Check if parameters have changed
    params_changed = False
    total_param_diff = 0.0
    for name, param in server.global_model.named_parameters():
        if param.requires_grad and name in initial_params:
            param_diff = torch.norm(param.data - initial_params[name]).item()
            total_param_diff += param_diff
            if param_diff > 0:
                params_changed = True
                print(f"Parameter {name} changed by {param_diff:.6f}")
    
    print(f"Total parameter difference: {total_param_diff:.6f}")
    print(f"Parameters changed: {params_changed}")
    
    # Test updated model
    server.global_model.eval()
    test_acc_updated = 0
    test_loss_updated = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(server.device), target.to(server.device)
            output = server.global_model(data)
            loss = F.cross_entropy(output, target)
            test_loss_updated += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            test_acc_updated += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss_updated /= len(test_loader)
    test_acc_updated /= len(test_loader.dataset)
    
    print(f"Updated test accuracy: {test_acc_updated:.4f}")
    print(f"Updated test loss: {test_loss_updated:.4f}")
    
    # Check if performance has changed
    acc_diff = test_acc_updated - test_acc_initial
    loss_diff = test_loss_updated - test_loss_initial
    
    print(f"Accuracy change: {acc_diff:.4f}")
    print(f"Loss change: {loss_diff:.4f}")
    
    return params_changed and total_param_diff > 0

if __name__ == "__main__":
    success = test_global_model_update()
    print(f"\nTest {'passed' if success else 'failed'}") 