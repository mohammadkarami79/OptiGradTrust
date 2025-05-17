import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.models.vae import VAE
from federated_learning.models.attention import DualAttention
from federated_learning.config.config import *
from federated_learning.data.dataset import load_dataset, split_dataset
from federated_learning.main import get_root_dataset, evaluate_model, pretrain_global_model, collect_root_gradients

def configure_gpu_logging():
    """Configure GPU logging to only print once."""
    if torch.cuda.is_available():
        print("\n=== GPU Configuration ===")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        
        # Print GPU memory usage
        print("\nGPU Memory:")
        print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def test_dataset_separation():
    """Test proper dataset separation into train, test, and root datasets."""
    print("\n=== Testing Dataset Separation ===")
    
    # Load datasets
    train_dataset, test_dataset, num_classes, input_channels = load_dataset()
    
    # Create root dataset
    root_dataset, root_indices = get_root_dataset(train_dataset)
    
    # Create client datasets from remaining data
    remaining_indices = list(set(range(len(train_dataset))) - set(root_indices))
    remaining_dataset = torch.utils.data.Subset(train_dataset, remaining_indices)
    
    # Split remaining dataset for clients
    client_datasets = split_dataset(remaining_dataset, num_classes)
    
    # Verify sizes
    print(f"Total train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Root dataset size: {len(root_dataset)}")
    print(f"Remaining dataset size: {len(remaining_dataset)}")
    
    # Verify client datasets
    total_client_samples = sum(len(ds) for ds in client_datasets)
    print(f"Number of clients: {len(client_datasets)}")
    print(f"Total samples across all clients: {total_client_samples}")
    
    # Verify no overlap between root and client datasets
    assert len(set(root_indices).intersection(set(remaining_indices))) == 0, "Root and client datasets overlap!"
    
    # Verify total samples match
    assert len(root_dataset) + len(remaining_dataset) == len(train_dataset), "Sample count mismatch!"
    
    print("Dataset separation test passed!")
    return train_dataset, test_dataset, root_dataset, client_datasets, num_classes

def test_vae_training(server, root_gradients):
    """Test VAE training on root gradients."""
    print("\n=== Testing VAE Training ===")
    
    # Train VAE
    server.train_vae(root_gradients)
    
    # Test reconstruction
    with torch.no_grad():
        server.vae.eval()
        
        # Get the device from VAE parameters
        vae_device = next(server.vae.parameters()).device
        
        # Test on a few root gradients
        for i, grad in enumerate(root_gradients[:3]):
            # Ensure gradient has batch dimension
            if grad.dim() == 1:
                grad = grad.unsqueeze(0)
            
            # Move to VAE device
            grad = grad.to(vae_device)
            
            # Reconstruct
            recon, mu, logvar = server.vae(grad)
            
            # Calculate reconstruction error
            recon_error = F.mse_loss(recon, grad).item()
            
            print(f"Gradient {i} - Reconstruction Error: {recon_error:.6f}")
            
            # Verify reasonable reconstruction error
            assert recon_error < 10.0, f"VAE reconstruction error too high: {recon_error}"
    
    print("VAE training test passed!")
    return server.vae

def test_dual_attention(server):
    """Test dual attention model with 5-feature vectors."""
    print("\n=== Testing Dual Attention Model ===")
    
    # Create test features with 5 dimensions
    num_clients = 10
    features = torch.rand((num_clients, 5), device=server.device)
    
    # Create global context
    global_context = features.mean(dim=0, keepdim=True)
    
    # Test forward pass
    trust_scores, confidence_scores = server.dual_attention(features, global_context)
    
    # Verify shapes
    assert trust_scores.shape == torch.Size([num_clients]), f"Trust scores shape mismatch: {trust_scores.shape}"
    assert confidence_scores.shape == torch.Size([num_clients]), f"Confidence scores shape mismatch: {confidence_scores.shape}"
    
    # Verify values are in [0, 1] range
    assert torch.all((trust_scores >= 0) & (trust_scores <= 1)), "Trust scores out of range!"
    assert torch.all((confidence_scores >= 0) & (confidence_scores <= 1)), "Confidence scores out of range!"
    
    # Test get_gradient_weights
    weights = server.dual_attention.get_gradient_weights(features, global_context)
    
    # Verify weights sum to 1
    assert abs(weights.sum().item() - 1.0) < 1e-5, f"Weights don't sum to 1: {weights.sum().item()}"
    
    print("Dual attention test passed!")
    return server.dual_attention

def test_gradient_feature_extraction(server, root_gradients):
    """Test gradient feature extraction with 5 standardized features."""
    print("\n=== Testing Gradient Feature Extraction ===")
    
    # Extract features from a few root gradients
    for i, grad in enumerate(root_gradients[:3]):
        # Compute features
        features = server._compute_gradient_features(
            grad=grad,
            raw_grad=grad,
            client_gradients=root_gradients,
            all_raw_gradients=root_gradients
        )
        
        # Verify shape
        assert features.shape == torch.Size([5]), f"Feature shape mismatch: {features.shape}"
        
        # Verify values are in [0, 1] range
        assert torch.all((features >= 0) & (features <= 1)), f"Features out of range: {features}"
        
        # Print features
        print(f"Gradient {i} Features:")
        print(f"  Reconstruction Error: {features[0]:.4f}")
        print(f"  Root Similarity: {features[1]:.4f}")
        print(f"  Client Similarity: {features[2]:.4f}")
        print(f"  Gradient Norm: {features[3]:.4f}")
        print(f"  Pattern Consistency: {features[4]:.4f}")
    
    print("Gradient feature extraction test passed!")

def test_gradient_aggregation(server, client_gradients):
    """Test gradient aggregation with dual attention."""
    print("\n=== Testing Gradient Aggregation ===")
    
    # Aggregate gradients
    aggregated_gradient = server.aggregate_gradients(client_gradients, None, 1)
    
    # Verify aggregated gradient is not None
    assert aggregated_gradient is not None, "Aggregated gradient is None!"
    
    # Verify shape matches client gradients
    assert aggregated_gradient.shape == client_gradients[0].shape, "Aggregated gradient shape mismatch!"
    
    # Verify weights are stored
    assert hasattr(server, 'weights'), "Weights not stored in server!"
    assert len(server.weights) == len(client_gradients), "Weights length mismatch!"
    
    print("Gradient aggregation test passed!")
    return aggregated_gradient

def test_global_model_update(server, aggregated_gradient):
    """Test global model update with aggregated gradient."""
    print("\n=== Testing Global Model Update ===")
    
    # Store initial parameters
    initial_params = {}
    for name, param in server.global_model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.clone().detach()
    
    # Update global model
    updated_model = server._update_global_model(aggregated_gradient, 1)
    
    # Verify model is updated
    assert updated_model is not None, "Updated model is None!"
    
    # Verify parameters changed
    params_changed = False
    for name, param in updated_model.named_parameters():
        if param.requires_grad and name in initial_params:
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                print(f"Parameter {name} changed")
                break
    
    assert params_changed, "No parameters changed after update!"
    
    print("Global model update test passed!")
    return updated_model

def test_complete_federated_learning_round(server, clients, test_loader):
    """Test a complete federated learning round."""
    print("\n=== Testing Complete Federated Learning Round ===")
    
    # Initial evaluation
    initial_error = evaluate_model(server.global_model, test_loader)
    print(f"Initial test error: {initial_error:.4f}")
    
    # Select clients
    num_selected = max(1, int(CLIENT_SELECTION_RATIO * len(clients)))
    selected_clients = random.sample(clients, num_selected)
    print(f"Selected {len(selected_clients)} clients")
    
    # Train selected clients
    client_gradients = []
    for client in selected_clients:
        print(f"Training client {client.client_id}...")
        gradient, _ = client.train(server.global_model, 1)
        if gradient is not None:
            client_gradients.append(gradient)
    
    # Aggregate gradients
    print("Aggregating gradients...")
    aggregated_gradient = server.aggregate_gradients(client_gradients, None, 1)
    
    # Update global model
    print("Updating global model...")
    updated_model = server._update_global_model(aggregated_gradient, 1)
    
    # Ensure the global model is updated
    server.global_model = updated_model
    
    # Final evaluation
    final_error = evaluate_model(server.global_model, test_loader)
    print(f"Final test error: {final_error:.4f}")
    
    print("Complete federated learning round test passed!")

def main():
    """Main test function."""
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Configure GPU logging
    configure_gpu_logging()
    
    print("\n=== Complete System Test ===")
    
    # Test dataset separation
    train_dataset, test_dataset, root_dataset, client_datasets, num_classes = test_dataset_separation()
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS if torch.cuda.is_available() else 0,
        pin_memory=PIN_MEMORY if torch.cuda.is_available() else False
    )
    
    # Create root loader
    root_loader = torch.utils.data.DataLoader(
        root_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS if torch.cuda.is_available() else 0,
        pin_memory=PIN_MEMORY if torch.cuda.is_available() else False
    )
    
    # Create server
    server = Server()
    server.root_loader = root_loader
    
    # Create clients
    clients = []
    for i in range(NUM_CLIENTS):
        is_malicious = i < NUM_MALICIOUS
        client = Client(i, client_datasets[i], is_malicious=is_malicious)
        clients.append(client)
    
    # Store clients in server
    server.clients = clients
    server.malicious_clients = list(range(NUM_MALICIOUS))
    
    # Pre-train global model on root dataset
    print("\n=== Pre-training global model on root dataset ===")
    pretrain_global_model(server.global_model, root_loader)
    
    # Collect root gradients
    print("\n=== Collecting root gradients ===")
    root_gradients = collect_root_gradients(copy.deepcopy(server.global_model), root_loader)
    server.root_gradients = root_gradients
    
    # Test VAE training
    vae = test_vae_training(server, root_gradients)
    
    # Test dual attention
    dual_attention = test_dual_attention(server)
    
    # Test gradient feature extraction
    test_gradient_feature_extraction(server, root_gradients)
    
    # Generate some client gradients for testing
    print("\n=== Generating client gradients for testing ===")
    client_gradients = []
    for i in range(5):
        # Create a random gradient with same shape as root gradients
        if i < NUM_MALICIOUS:
            # For malicious clients, create adversarial gradients
            if ATTACK_TYPE == 'label_flipping':
                # Reverse direction
                grad = -root_gradients[0].clone()
            elif ATTACK_TYPE == 'scaling_attack':
                # Scale up
                grad = root_gradients[0].clone() * 3.0
            else:
                # Add noise
                grad = root_gradients[0].clone() + torch.randn_like(root_gradients[0]) * 0.5
        else:
            # For honest clients, use root gradients with small variations
            grad = root_gradients[0].clone() + torch.randn_like(root_gradients[0]) * 0.1
        
        client_gradients.append(grad)
    
    # Test gradient aggregation
    aggregated_gradient = test_gradient_aggregation(server, client_gradients)
    
    # Test global model update
    updated_model = test_global_model_update(server, aggregated_gradient)
    
    # Test complete federated learning round
    test_complete_federated_learning_round(server, clients, test_loader)
    
    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    main() 