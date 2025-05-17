import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics
from federated_learning.config.config import *
import psutil
import os
import copy
import math
from federated_learning.models.attention import DualAttention

def test(model, test_loader):
    """
    Test the model on the test dataset.
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
        
    Returns:
        accuracy: Test accuracy
        error_rate: Test error rate
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Get the device from the model
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / total
    error_rate = 1 - accuracy
    
    print(f"Test Accuracy: {accuracy:.4f}, Error Rate: {error_rate:.4f}")
    
    return accuracy, error_rate

def client_update(client_model, optimizer, train_loader, epochs, global_model=None):
    """Update client model using local data"""
    device = next(client_model.parameters()).device
    client_model.train()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    
    return client_model

def train_vae(vae, gradient_stack, epochs=5, batch_size=32, learning_rate=1e-3, device=None):
    """Train a VAE model on gradients"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vae = vae.to(device)
    gradient_stack = gradient_stack.to(device)
    
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(gradient_stack), batch_size):
            batch = gradient_stack[i:i+batch_size]
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = vae(batch)
            
            # Calculate loss
            loss = vae_loss(recon_batch, batch, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / (len(gradient_stack) / batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return vae

def get_process_memory_usage():
    """Get the current memory usage of this process in GB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)
    return memory_gb

def vae_loss(recon_x, x, mu, logvar):
    """Calculate the VAE loss function"""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + 0.1 * kld

def train_dual_attention(honest_features, malicious_features=None, epochs=100, 
                        batch_size=32, lr=0.001, weight_decay=1e-4,
                    device=None, verbose=True, early_stopping=10):
    """
    Train the dual attention model for malicious client detection.
    
    Args:
        honest_features: Tensor of feature vectors from honest clients [num_honest, feature_dim]
        malicious_features: Tensor of feature vectors from malicious clients [num_malicious, feature_dim]
                           If None, malicious features will be generated
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        verbose: Whether to print training progress
        early_stopping: Number of epochs without improvement before stopping
        
    Returns:
        model: Trained DualAttention model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if we have honest features
    if honest_features is None or len(honest_features) == 0:
        raise ValueError("Must provide honest features for training")
    
    # Convert to tensors if needed
    if not isinstance(honest_features, torch.Tensor):
        honest_features = torch.tensor(honest_features, dtype=torch.float32)
    
    # Handle NaN values
    if torch.isnan(honest_features).any():
        print("Warning: NaN values in honest features, replacing with zeros")
        honest_features = torch.nan_to_num(honest_features, nan=0.0)
    
    # Move to device
    honest_features = honest_features.to(device)
    
    # Determine feature dimension
    feature_dim = honest_features.shape[1]
    print(f"Feature dimension for DualAttention training: {feature_dim}")
    
    # Print honest feature statistics for debugging
    print("\nHonest feature statistics:")
    honest_means = []
    honest_stds = []
    for i in range(feature_dim):
        mean_val = honest_features[:, i].mean().item()
        std_val = honest_features[:, i].std().item()
        min_val = honest_features[:, i].min().item()
        max_val = honest_features[:, i].max().item()
        honest_means.append(mean_val)
        honest_stds.append(std_val)
        print(f"Feature {i+1}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
        print(f"  Min = {min_val:.4f}, Max = {max_val:.4f}")
        
        # Check for constant features and fix if needed
        if std_val < 1e-5:
            print(f"  WARNING: Feature {i+1} has near-zero variance. Adding noise for training stability.")
            noise = torch.randn_like(honest_features[:, i]) * 0.05
            honest_features[:, i] = torch.clamp(honest_features[:, i] + noise, 0.0, 1.0)
    
    # Generate or process malicious features
    if malicious_features is None:
        print("Generating synthetic malicious features using multiple attack types")
        # Synthetic generation of malicious features with clear differences
        num_synthetic = min(max(honest_features.shape[0] * 2, 100), 500)  # Generate enough samples
        malicious_features = torch.zeros((num_synthetic, feature_dim), device=device)
        
        # For each feature, make it distinctly different from honest features
        for i in range(feature_dim):
            if i == 0:  # VAE Reconstruction Error - higher for malicious
                malicious_features[:, i] = torch.clamp(
                    torch.normal(mean=min(honest_means[i] * 3, 0.7), std=0.2, size=(num_synthetic,), device=device),
                    0.05, 0.95
                )
            elif i == 1:  # Root Similarity - lower for malicious
                malicious_features[:, i] = torch.clamp(
                    torch.normal(mean=max(honest_means[i] * 0.5, 0.2), std=0.15, size=(num_synthetic,), device=device),
                    0.05, 0.95
                )
            elif i == 2:  # Client Similarity - lower for malicious
                malicious_features[:, i] = torch.clamp(
                    torch.normal(mean=max(honest_means[i] * 0.5, 0.2), std=0.15, size=(num_synthetic,), device=device),
                    0.05, 0.95
                )
            elif i == 3:  # Gradient Norm - much higher for malicious
                malicious_features[:, i] = torch.clamp(
                    torch.normal(mean=min(honest_means[i] * 5, 0.8), std=0.2, size=(num_synthetic,), device=device),
                    0.1, 0.99
                )
            elif i == 4:  # Sign Consistency - lower for malicious
                malicious_features[:, i] = torch.clamp(
                    torch.normal(mean=max(honest_means[i] * 0.6, 0.3), std=0.15, size=(num_synthetic,), device=device),
                    0.05, 0.95
                )
            elif i == 5 and feature_dim > 5:  # Shapley Value - higher for malicious (more influence)
                malicious_features[:, i] = torch.clamp(
                    torch.normal(mean=min(honest_means[i] * 2, 0.8), std=0.2, size=(num_synthetic,), device=device),
                    0.1, 0.99
                )
        
        print(f"Generated {num_synthetic} synthetic malicious feature vectors")
    else:
        # Use provided malicious features
        if not isinstance(malicious_features, torch.Tensor):
            malicious_features = torch.tensor(malicious_features, dtype=torch.float32)
            
        if torch.isnan(malicious_features).any():
            print("Warning: NaN values in malicious features, replacing with zeros")
            malicious_features = torch.nan_to_num(malicious_features, nan=0.0)
            
        malicious_features = malicious_features.to(device)
        
        # Check dimension consistency
        if malicious_features.shape[1] != feature_dim:
            print(f"Warning: Malicious feature dimension ({malicious_features.shape[1]}) doesn't match honest features ({feature_dim})")
            # Adjust if possible
            if malicious_features.shape[1] < feature_dim:
                # Pad with zeros
                padding = torch.zeros(malicious_features.shape[0], feature_dim - malicious_features.shape[1], device=device)
                malicious_features = torch.cat([malicious_features, padding], dim=1)
                print(f"Padded malicious features to shape: {malicious_features.shape}")
            else:
                # Truncate
                malicious_features = malicious_features[:, :feature_dim]
                print(f"Truncated malicious features to shape: {malicious_features.shape}")
                
        # Add noise to constant features
        for i in range(feature_dim):
            if malicious_features[:, i].std() < 1e-5:
                print(f"  WARNING: Malicious feature {i+1} has near-zero variance. Adding noise for training stability.")
                noise = torch.randn_like(malicious_features[:, i]) * 0.05
                malicious_features[:, i] = torch.clamp(malicious_features[:, i] + noise, 0.0, 1.0)
    
    # Print malicious feature statistics for comparison
    print("\nMalicious feature statistics:")
    for i in range(feature_dim):
        mean_val = malicious_features[:, i].mean().item()
        std_val = malicious_features[:, i].std().item()
        min_val = malicious_features[:, i].min().item()
        max_val = malicious_features[:, i].max().item()
        honest_mean = honest_features[:, i].mean().item()
        
        # Calculate difference and percentage change
        diff = mean_val - honest_mean
        pct_change = (diff / max(honest_mean, 1e-5)) * 100 if honest_mean != 0 else float('inf')
        
        print(f"Feature {i+1}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
        print(f"  Min = {min_val:.4f}, Max = {max_val:.4f}")
        print(f"  Difference from honest: {diff:.4f} ({pct_change:.1f}%)")
    
    # Ensure we have enough diversity in the dataset by enhanced feature differentiation
    # This helps to prevent the model from classifying everything as one class
    feature_diff_sum = sum(abs(malicious_features.mean(dim=0) - honest_features.mean(dim=0)))
    print(f"Total feature differentiation: {feature_diff_sum.item():.4f}")
    
    if feature_diff_sum < 0.5:
        print("WARNING: Low feature differentiation between honest and malicious samples.")
        print("Enhancing feature contrast to improve model training...")
        
        # Enhance gradient norm difference (most important feature)
        if feature_dim > 3:
            honest_norm_mean = honest_features[:, 3].mean().item()
            malicious_features[:, 3] = torch.clamp(malicious_features[:, 3] * 1.5, honest_norm_mean + 0.2, 0.99)
    
    # Create labels
    honest_labels = torch.zeros(honest_features.shape[0], device=device)
    malicious_labels = torch.ones(malicious_features.shape[0], device=device)
    
    # Combine features and labels
    all_features = torch.cat([honest_features, malicious_features], dim=0)
    all_labels = torch.cat([honest_labels, malicious_labels], dim=0)
    
    # Calculate class weights to handle imbalance
    num_honest = len(honest_labels)
    num_malicious = len(malicious_labels)
    total_samples = num_honest + num_malicious
    
    # Create sample weights for weighted loss
    weight_honest = 1.0 if num_malicious == 0 else total_samples / (2 * num_honest)
    weight_malicious = 1.0 if num_honest == 0 else total_samples / (2 * num_malicious)
    print(f"Class weights - Honest: {weight_honest:.4f}, Malicious: {weight_malicious:.4f}")
    
    # Create sample weights tensor
    sample_weights = torch.where(all_labels == 0, 
                                torch.tensor(weight_honest, device=device),
                                torch.tensor(weight_malicious, device=device))
    
    # Create dataset with sample weights
    dataset = TensorDataset(all_features, all_labels, sample_weights)
    
    # Split into train and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create DualAttention model
    from federated_learning.models.attention import DualAttention
    model = DualAttention(
        feature_dim=feature_dim,
        hidden_dim=DUAL_ATTENTION_HIDDEN_SIZE,
        num_heads=DUAL_ATTENTION_HEADS,
        num_layers=DUAL_ATTENTION_LAYERS,
        dropout=0.2
    ).to(device)
    
    # Print model structure
    print(f"Created DualAttention model with feature_dim={feature_dim}")
    
    # Training setup with improved components
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=verbose)
    
    # Custom weighted BCE loss for imbalanced classes
    def weighted_bce_loss(predictions, targets, weights):
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        weighted_loss = bce_loss * weights
        return weighted_loss.mean()
    
    # Training loop with validation
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    no_improve_epochs = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_true_pos = 0
        train_pred_pos = 0
        train_actual_pos = 0
        
        for batch_idx, (features, labels, weights) in enumerate(train_loader):
            # Forward pass
            try:
                # Create global context from honest features
                honest_mask = (labels == 0)
                if honest_mask.sum() > 0:
                    honest_context = features[honest_mask].mean(dim=0, keepdim=True)
                else:
                    honest_context = features.mean(dim=0, keepdim=True)
                
                # Get predictions and confidence
                scores, confidence = model(features, honest_context)
                
                # Ensure scores are properly clamped to avoid numerical issues
                scores = torch.clamp(scores, 1e-7, 1-1e-7)
                
                # Weighted BCE loss
                loss = weighted_bce_loss(scores, labels, weights)
                
                # Add separation/margin loss to make malicious clearly different from honest
                honest_scores = scores[labels == 0]
                malicious_scores = scores[labels == 1]
                
                if len(honest_scores) > 0 and len(malicious_scores) > 0:
                    margin_target = 0.5  # We want at least this much separation
                    actual_margin = torch.mean(malicious_scores) - torch.mean(honest_scores)
                    margin_loss = F.relu(margin_target - actual_margin)
                    loss = loss + 0.1 * margin_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * features.size(0)
                predicted = (scores > 0.5).float()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # For F1 score calculation
                train_true_pos += ((predicted == 1) & (labels == 1)).sum().item()
                train_pred_pos += (predicted == 1).sum().item()
                train_actual_pos += (labels == 1).sum().item()
                
            except Exception as e:
                print(f"Error in training batch: {str(e)}")
                continue
        
        # Calculate training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Calculate F1 score components
        train_precision = train_true_pos / max(train_pred_pos, 1)
        train_recall = train_true_pos / max(train_actual_pos, 1)
        train_f1 = 2 * (train_precision * train_recall) / max((train_precision + train_recall), 1e-8)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_true_pos = 0
        val_pred_pos = 0
        val_actual_pos = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (features, labels, weights) in enumerate(val_loader):
                # Create validation context
                honest_mask = (labels == 0)
                if honest_mask.sum() > 0:
                    honest_context = features[honest_mask].mean(dim=0, keepdim=True)
                else:
                    honest_context = features.mean(dim=0, keepdim=True)
                
                # Forward pass
                scores, _ = model(features, honest_context)
                scores = torch.clamp(scores, 1e-7, 1-1e-7)
                
                # Loss
                loss = weighted_bce_loss(scores, labels, weights)
                val_loss += loss.item() * features.size(0)
                
                # Track metrics
                predicted = (scores > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # For detailed metrics
                val_true_pos += ((predicted == 1) & (labels == 1)).sum().item()
                val_pred_pos += (predicted == 1).sum().item()
                val_actual_pos += (labels == 1).sum().item()
                
                # Save for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Calculate F1 score components
        val_precision = val_true_pos / max(val_pred_pos, 1)
        val_recall = val_true_pos / max(val_actual_pos, 1)
        val_f1 = 2 * (val_precision * val_recall) / max((val_precision + val_recall), 1e-8)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save metrics for plotting
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"  Train - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
            print(f"  Val - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Check if this is the best model so far
        # We use F1 score for imbalanced datasets as it's a better metric
        if val_f1 > best_val_acc:
            best_val_acc = val_f1
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Early stopping
        if no_improve_epochs >= early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on all data
    model.eval()
    with torch.no_grad():
        # Evaluate on all data
        all_preds = []
        scores, _ = model(all_features)
        predicted = (scores > 0.5).float()
        accuracy = (predicted == all_labels).float().mean().item()
        
        # Calculate metrics for honest and malicious separately
        honest_mask = all_labels == 0
        malicious_mask = all_labels == 1
        
        # Class-specific accuracy
        honest_acc = (predicted[honest_mask] == all_labels[honest_mask]).float().mean().item() if honest_mask.any() else 0
        malicious_acc = (predicted[malicious_mask] == all_labels[malicious_mask]).float().mean().item() if malicious_mask.any() else 0
        
        # True positives, false positives, etc.
        true_pos = ((predicted == 1) & (all_labels == 1)).sum().item()
        false_pos = ((predicted == 1) & (all_labels == 0)).sum().item()
        false_neg = ((predicted == 0) & (all_labels == 1)).sum().item()
        
        # Calculate precision, recall, F1
        precision = true_pos / max(true_pos + false_pos, 1)
        recall = true_pos / max(true_pos + false_neg, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
        
        print(f"\nFinal model performance - Overall accuracy: {accuracy:.4f}")
        print(f"Honest client detection accuracy: {honest_acc:.4f}")
        print(f"Malicious client detection accuracy: {malicious_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(f"True Positives: {true_pos}, False Positives: {false_pos}")
        print(f"False Negatives: {false_neg}, True Negatives: {(honest_mask & (predicted == 0)).sum().item()}")
    
    # Save model for future use
    try:
        model_dir = 'model_weights'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, 'dual_attention.pth'))
        print(f"Saved trained dual attention model to {os.path.join(model_dir, 'dual_attention.pth')}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    return model