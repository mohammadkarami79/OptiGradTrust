"""
Training utility functions for federated learning.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from federated_learning.config.config import *

def evaluate_model(model, test_loader):
    """
    Evaluate model accuracy on the test dataset.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        
    Returns:
        error: Error rate (1 - accuracy)
    """
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
    
    error = 1.0 - correct / total
    return error

def test(model, test_loader):
    """
    Test the model and return accuracy and error rate.
    
    Args:
        model: Model to test
        test_loader: Test data loader
        
    Returns:
        accuracy: Accuracy rate
        error: Error rate
    """
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

def collect_root_gradients(model, root_loader, num_epochs=LOCAL_EPOCHS_ROOT):
    """
    Collect gradients from training on the root dataset for VAE training.
    
    Args:
        model: Model to train
        root_loader: Root data loader
        num_epochs: Number of epochs to train
        
    Returns:
        root_gradients: List of gradients from root training
    """
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

def pretrain_global_model(model, train_loader):
    """
    Pre-train the global model on the root dataset.
    
    Args:
        model: Model to train
        train_loader: Training data loader
    """
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

def train_vae(vae, gradient_stack, epochs=VAE_EPOCHS, batch_size=VAE_BATCH_SIZE, learning_rate=VAE_LEARNING_RATE, device=None):
    """
    Train VAE on collected gradients.
    
    Args:
        vae: VAE model
        gradient_stack: Stack of gradients
        epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        vae: Trained VAE model
    """
    if device is None:
        device = next(vae.parameters()).device
    
    vae.train()
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    # Create dataset from gradient stack
    dataset = torch.utils.data.TensorDataset(gradient_stack)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = vae(data)
            
            # Calculate loss
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
        
        # Print epoch statistics
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        print(f"VAE Epoch {epoch+1}/{epochs}, "
              f"Loss: {avg_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, "
              f"KL Loss: {avg_kl_loss:.4f}")
    
    print("VAE training completed")
    return vae

def train_dual_attention(dual_attention, features, labels, epochs=DUAL_ATTENTION_EPOCHS, batch_size=DUAL_ATTENTION_BATCH_SIZE, learning_rate=DUAL_ATTENTION_LEARNING_RATE):
    """
    Train dual attention model.
    
    Args:
        dual_attention: Dual attention model
        features: Feature vectors
        labels: Labels (0=honest, 1=malicious)
        epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        dual_attention: Trained dual attention model
    """
    device = next(dual_attention.parameters()).device
    dual_attention.train()
    
    # Move data to device
    features = features.to(device)
    labels = labels.to(device)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(features, labels)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Configure optimizer
    optimizer = optim.Adam(dual_attention.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        dual_attention.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Global context from mean of features
            global_context = data.mean(dim=0, keepdim=True)
            
            # Forward pass
            trust_scores, _ = dual_attention(data, global_context)
            
            # Calculate loss
            loss = F.binary_cross_entropy(trust_scores, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            train_preds = (trust_scores > 0.5).float()
            train_correct += (train_preds == target).sum().item()
            train_total += target.size(0)
        
        # Validation
        dual_attention.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                
                # Global context from mean of features
                global_context = data.mean(dim=0, keepdim=True)
                
                # Forward pass
                trust_scores, _ = dual_attention(data, global_context)
                
                # Calculate loss
                loss = F.binary_cross_entropy(trust_scores, target)
                
                # Track statistics
                val_loss += loss.item()
                val_preds = (trust_scores > 0.5).float()
                val_correct += (val_preds == target).sum().item()
                val_total += target.size(0)
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(dual_attention.state_dict(), 'best_dual_attention.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                # Load best model
                dual_attention.load_state_dict(torch.load('best_dual_attention.pth'))
                break
    
    print("Dual attention training completed")
    return dual_attention 