import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
import os
import matplotlib.pyplot as plt
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.models.vae import VAE
from federated_learning.models.attention import DualAttention
from federated_learning.config.config import *
from federated_learning.data.dataset import load_dataset, split_dataset
from federated_learning.utils.gradient_features import compute_gradient_features, normalize_features
from federated_learning.utils.model_utils import update_model_with_gradient, fine_tune_model, adaptive_learning_rate

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

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_root_dataset(train_dataset):
    """Create root dataset from training dataset."""
    print("\n=== Creating Root Dataset ===")
    
    # Determine root dataset size
    if ROOT_DATASET_DYNAMIC_SIZE:
        root_size = int(len(train_dataset) * ROOT_DATASET_RATIO)
    else:
        root_size = min(ROOT_DATASET_SIZE, len(train_dataset))
    
    print(f"Root dataset size: {root_size} samples")
    
    # Create indices for root dataset
    all_indices = list(range(len(train_dataset)))
    root_indices = random.sample(all_indices, root_size)
    
    # Create root dataset
    root_dataset = torch.utils.data.Subset(train_dataset, root_indices)
    
    return root_dataset, root_indices

def evaluate_model(model, test_loader):
    """Evaluate model accuracy on the test dataset."""
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    error = 1.0 - accuracy
    return accuracy, error

def pretrain_global_model(model, train_loader):
    """Pre-train the global model on the root dataset."""
    print("\n=== Pre-training Global Model ===")
    device = next(model.parameters()).device
    model.train()
    
    # Configure optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Train for specified number of epochs
    for epoch in range(LOCAL_EPOCHS_ROOT):
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            if batch_idx % 50 == 0:
                print(f"Pretrain Epoch {epoch+1}/{LOCAL_EPOCHS_ROOT}, "
                      f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calculate average loss for this epoch
        epoch_loss = running_loss / total_samples
        print(f"Pretrain Epoch {epoch+1}/{LOCAL_EPOCHS_ROOT} completed. "
              f"Average loss: {epoch_loss:.4f}")
    
    print("Pretraining completed")
    return model

def collect_root_gradients(model, root_loader, num_epochs=LOCAL_EPOCHS_ROOT):
    """Collect gradients from training on the root dataset for VAE training."""
    print("\n=== Collecting Root Gradients ===")
    device = next(model.parameters()).device
    model.train()
    root_gradients = []
    
    # Create a copy of the initial model to compute gradients against
    initial_model = copy.deepcopy(model)
    
    # Configure optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    for epoch in range(num_epochs):
        print(f"\nRoot Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        for batch_idx, (data, target) in enumerate(root_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f"Root Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # After each epoch, collect the gradient (difference between current and initial model)
        grad_list = []
        for (name, p_current), (_, p_initial) in zip(model.named_parameters(), initial_model.named_parameters()):
            if p_current.requires_grad:
                # Compute the difference
                diff = (p_current.data - p_initial.data).view(-1)
                grad_list.append(diff)
        
        # Concatenate all parameter differences
        if grad_list:
            gradient = torch.cat(grad_list).detach()
            root_gradients.append(gradient)
            print(f"Collected root gradient {epoch+1}, norm: {torch.norm(gradient).item():.4f}")
    
    print(f"Collected {len(root_gradients)} root gradients")
    return root_gradients

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
        features = compute_gradient_features(
            grad=grad,
            raw_grad=grad,
            vae=server.vae,
            root_gradients=root_gradients,
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
        
        # Normalize features
        normalized_features = normalize_features(features.unsqueeze(0)).squeeze(0)
        
        # Verify normalization preserves range
        assert torch.all((normalized_features >= 0) & (normalized_features <= 1)), \
            f"Normalized features out of range: {normalized_features}"
        
        print(f"  After Normalization:")
        print(f"  Reconstruction Error: {normalized_features[0]:.4f}")
        print(f"  Root Similarity: {normalized_features[1]:.4f}")
        print(f"  Client Similarity: {normalized_features[2]:.4f}")
        print(f"  Gradient Norm: {normalized_features[3]:.4f}")
        print(f"  Pattern Consistency: {normalized_features[4]:.4f}")
    
    # Test batch normalization
    batch_features = torch.stack([
        compute_gradient_features(
            grad=grad,
            raw_grad=grad,
            vae=server.vae,
            root_gradients=root_gradients,
            client_gradients=root_gradients,
            all_raw_gradients=root_gradients
        ) for grad in root_gradients[:5]
    ])
    
    # Normalize batch
    normalized_batch = normalize_features(batch_features)
    
    # Verify batch normalization preserves range
    assert torch.all((normalized_batch >= 0) & (normalized_batch <= 1)), \
        f"Batch normalized features out of range"
    
    # Verify batch shape is preserved
    assert normalized_batch.shape == batch_features.shape, \
        f"Batch shape changed during normalization"
    
    print("Gradient feature extraction test passed!")

def test_dual_attention_training(server, root_gradients):
    """Test dual attention training with both honest and malicious gradients."""
    print("\n=== Testing Dual Attention Training ===")
    
    # Create honest features from root gradients
    honest_features = []
    for grad in root_gradients[:5]:
        features = compute_gradient_features(
            grad=grad,
            raw_grad=grad,
            vae=server.vae,
            root_gradients=root_gradients,
            client_gradients=root_gradients,
            all_raw_gradients=root_gradients
        )
        honest_features.append(features)
    
    # Create malicious features (simulated attacks)
    malicious_features = []
    base_grad = root_gradients[0]
    
    # 1. Scaling attack
    scaled_grad = base_grad * 5.0
    scaled_grad = scaled_grad / torch.norm(scaled_grad)
    features = compute_gradient_features(
        grad=scaled_grad,
        raw_grad=scaled_grad,
        vae=server.vae,
        root_gradients=root_gradients,
        client_gradients=root_gradients,
        all_raw_gradients=root_gradients
    )
    malicious_features.append(features)
    
    # 2. Sign flipping attack
    flipped_grad = -base_grad
    features = compute_gradient_features(
        grad=flipped_grad,
        raw_grad=flipped_grad,
        vae=server.vae,
        root_gradients=root_gradients,
        client_gradients=root_gradients,
        all_raw_gradients=root_gradients
    )
    malicious_features.append(features)
    
    # 3. Random noise attack
    noise_grad = torch.randn_like(base_grad)
    noise_grad = noise_grad / torch.norm(noise_grad)
    features = compute_gradient_features(
        grad=noise_grad,
        raw_grad=noise_grad,
        vae=server.vae,
        root_gradients=root_gradients,
        client_gradients=root_gradients,
        all_raw_gradients=root_gradients
    )
    malicious_features.append(features)
    
    # Stack features
    honest_features = torch.stack(honest_features)
    malicious_features = torch.stack(malicious_features)
    
    # Print feature statistics
    print("\nHonest Feature Statistics:")
    honest_mean = honest_features.mean(dim=0)
    honest_std = honest_features.std(dim=0)
    feature_names = ["RE", "Root Sim", "Client Sim", "Norm", "Consistency"]
    for i, name in enumerate(feature_names):
        print(f"{name}: {honest_mean[i]:.4f} ± {honest_std[i]:.4f}")
    
    print("\nMalicious Feature Statistics:")
    malicious_mean = malicious_features.mean(dim=0)
    malicious_std = malicious_features.std(dim=0)
    for i, name in enumerate(feature_names):
        print(f"{name}: {malicious_mean[i]:.4f} ± {malicious_std[i]:.4f}")
    
    # Test dual attention with these features
    all_features = torch.cat([honest_features, malicious_features])
    labels = torch.cat([
        torch.zeros(len(honest_features)),
        torch.ones(len(malicious_features))
    ])
    
    # Move to device
    device = next(server.dual_attention.parameters()).device
    all_features = all_features.to(device)
    labels = labels.to(device)
    
    # Create global context
    global_context = all_features.mean(dim=0, keepdim=True)
    
    # Forward pass
    trust_scores, confidence_scores = server.dual_attention(all_features, global_context)
    
    # Calculate accuracy
    predictions = (trust_scores > 0.5).float()
    accuracy = (predictions == labels).float().mean().item()
    
    # Calculate class-specific metrics
    honest_indices = (labels == 0)
    malicious_indices = (labels == 1)
    
    honest_acc = (predictions[honest_indices] == labels[honest_indices]).float().mean().item()
    malicious_acc = (predictions[malicious_indices] == labels[malicious_indices]).float().mean().item()
    
    # Print results
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Honest Accuracy: {honest_acc:.4f}")
    print(f"Malicious Accuracy: {malicious_acc:.4f}")
    
    # Test gradient weighting
    weights = server.dual_attention.get_gradient_weights(all_features, global_context)
    
    honest_weights = weights[honest_indices].mean().item()
    malicious_weights = weights[malicious_indices].mean().item()
    
    print(f"Average Honest Weight: {honest_weights:.4f}")
    print(f"Average Malicious Weight: {malicious_weights:.4f}")
    print(f"Weight Ratio (Honest/Malicious): {honest_weights/max(malicious_weights, 1e-8):.4f}")
    
    # Verify weights sum to 1
    assert abs(weights.sum().item() - 1.0) < 1e-5, f"Weights don't sum to 1: {weights.sum().item()}"
    
    # Verify honest weights are higher than malicious weights
    if honest_weights <= malicious_weights:
        print("Warning: Honest weights are not higher than malicious weights!")
    
    print("Dual attention training test passed!")

def test_gradient_aggregation(server, client_gradients):
    """Test gradient aggregation with dual attention."""
    print("\n=== Testing Gradient Aggregation ===")
    
    # Create feature vectors for each gradient
    features_list = []
    for grad in client_gradients:
        features = compute_gradient_features(
            grad=grad,
            raw_grad=grad,
            vae=server.vae,
            root_gradients=server.root_gradients,
            client_gradients=client_gradients,
            all_raw_gradients=client_gradients
        )
        features_list.append(features)
    
    # Stack features into a tensor
    features_tensor = torch.stack(features_list)
    
    # Aggregate gradients
    aggregated_gradient = server.aggregate_gradients(client_gradients, features_tensor, 1)
    
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
    initial_acc, initial_error = evaluate_model(server.global_model, test_loader)
    print(f"Initial test accuracy: {initial_acc:.4f}")
    print(f"Initial test error: {initial_error:.4f}")
    
    # Select clients
    num_selected = max(1, int(CLIENT_SELECTION_RATIO * len(clients)))
    selected_clients = random.sample(clients, num_selected)
    print(f"Selected {len(selected_clients)} clients")
    
    # Train selected clients
    client_gradients = []
    client_ids = []
    features_list = []
    
    for client in selected_clients:
        print(f"Training client {client.client_id}...")
        gradient, features = client.train(server.global_model, 1)
        if gradient is not None:
            client_gradients.append(gradient)
            client_ids.append(client.client_id)
            
            # Extract features if not provided
            if features is None:
                features = compute_gradient_features(
                    grad=gradient,
                    raw_grad=gradient,
                    vae=server.vae,
                    root_gradients=server.root_gradients,
                    client_gradients=client_gradients,
                    all_raw_gradients=client_gradients
                )
            features_list.append(features)
    
    # Stack features into a tensor
    features_tensor = torch.stack(features_list)
    
    # Aggregate gradients
    print("Aggregating gradients...")
    aggregated_gradient = server.aggregate_gradients(client_gradients, features_tensor, 1)
    
    # Update global model
    print("Updating global model...")
    server.global_model = server._update_global_model(aggregated_gradient, 1)
    
    # Final evaluation
    final_acc, final_error = evaluate_model(server.global_model, test_loader)
    print(f"Final test accuracy: {final_acc:.4f}")
    print(f"Final test error: {final_error:.4f}")
    
    print("Complete federated learning round test passed!")

def main():
    """Main test function."""
    # Set random seed for reproducibility
    set_seed(SEED)
    
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
        num_workers=0,
        pin_memory=False
    )
    
    # Create root loader
    root_loader = torch.utils.data.DataLoader(
        root_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Create server
    server = Server()
    server.root_loader = root_loader
    server.test_dataset = test_dataset
    
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
    
    # Test enhanced gradient feature extraction
    test_gradient_feature_extraction(server, root_gradients)
    
    # Test dual attention
    dual_attention = test_dual_attention(server)
    
    # Test dual attention training
    test_dual_attention_training(server, root_gradients)
    
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