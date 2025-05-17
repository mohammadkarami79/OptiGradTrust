"""
Utility functions for model updates and training.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from federated_learning.config.config import *
import copy
import torch.nn as nn
import numpy as np
import random
import os

def update_model_with_gradient(model, gradient, learning_rate=None, proximal_mu=0.0, preserve_bn=False):
    """
    Update model parameters with the given gradient.
    
    Args:
        model: Model to update
        gradient: Gradient to apply
        learning_rate: Learning rate for update (uses config.LR if None)
        proximal_mu: Proximal term coefficient (for FedProx)
        preserve_bn: Whether to preserve BatchNorm parameters (for FedBN)
        
    Returns:
        updated_model: Updated model
        total_change: Total parameter change
        avg_change: Average parameter change
    """
    # Use default learning rate from config if not provided
    if learning_rate is None:
        learning_rate = LR
    
    # Debug: check input parameters
    grad_norm = torch.norm(gradient).item()
    print(f"Updating model with gradient (norm: {grad_norm:.8f}, learning rate: {learning_rate:.8f})")
    
    # Verify gradient is non-zero
    if grad_norm < 1e-10:
        print("WARNING: Gradient norm is nearly zero! This will result in no parameter updates.")
    
    # Verify gradient shape matches model parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if gradient.numel() != param_count:
        raise ValueError(f"Gradient size ({gradient.numel()}) doesn't match model parameters ({param_count})")
    
    # Get model device for memory optimization
    device = next(model.parameters()).device
    
    # Make sure gradient is on the same device as model
    if gradient.device != device:
        gradient = gradient.to(device)
    
    # Extract BatchNorm parameters if needed for FedBN
    bn_params = {}
    if preserve_bn:
        # Store original BatchNorm parameters
        bn_params = extract_bn_parameters(model)
        if bn_params:
            print(f"FedBN: Preserving {len(bn_params)} BatchNorm parameters")
    
    # Track total parameter change
    total_change = 0.0
    total_params = 0
    
    # Apply gradient to model parameters
    with torch.no_grad():
        idx = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Skip BatchNorm parameters if using FedBN
            if preserve_bn and any(bn_name == name for bn_name in bn_params):
                print(f"FedBN: Skipping update for BatchNorm parameter {name}")
                continue
            
            # Calculate start and end indices for this parameter
            size = param.numel()
            start, end = idx, idx + size
            idx = end
            
            # Get gradient for this parameter
            param_grad = gradient[start:end].reshape(param.shape)
            
            # Store original parameter for change calculation
            old_param = param.clone()
            
            # Update parameter: θ = θ - η*∇f(θ)
            if proximal_mu > 0.0:
                # FedProx: add proximal term: θ = θ - η*(∇f(θ) + μ*θ)
                param.data -= learning_rate * (param_grad + proximal_mu * param.data)
            else:
                # Standard update
                param.data -= learning_rate * param_grad
            
            # Calculate parameter change
            param_change = torch.sum(torch.abs(old_param - param)).item()
            total_change += param_change
            total_params += size
            
            # Check for zero change despite non-zero gradient
            param_grad_norm = torch.norm(param_grad).item()
            if param_change < 1e-10 and param_grad_norm > 1e-8:
                print(f"WARNING: No change in parameter '{name}' despite gradient norm {param_grad_norm:.8f}")
                # Try direct update with explicit casting
                param.data = param.data - learning_rate * param_grad
                # Verify change
                new_change = torch.sum(torch.abs(old_param - param)).item()
                if new_change > 1e-10:
                    print(f"Forced update successful: {new_change:.8f}")
                    total_change += new_change - param_change  # Adjust total change
    
    # Restore BatchNorm parameters if using FedBN
    if preserve_bn:
        for name, saved_param in bn_params.items():
            for n, p in model.named_parameters():
                if n == name:
                    # Restore the saved BatchNorm parameter
                    p.data.copy_(saved_param)
                    print(f"FedBN: Restored BatchNorm parameter {name}")
                    break
    
    # Calculate average change
    avg_change = total_change / max(1, total_params)
    
    return model, total_change, avg_change

def save_model(model, path):
    """
    Save a model to a file.
    
    Args:
        model: The model to save
        path: The path to save the model to
    """
    try:
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
        return True
    except Exception as e:
        print(f"Error saving model to {path}: {str(e)}")
        return False
        
def load_model(model, path):
    """
    Load a model from a file.
    
    Args:
        model: The model to load into
        path: The path to load the model from
        
    Returns:
        The loaded model
    """
    try:
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {str(e)}")
        return None

def fine_tune_model(model, data_loader, learning_rate=None, epochs=1):
    """
    Fine-tune a model on a dataset.
    
    Args:
        model: Model to fine-tune
        data_loader: Data loader for fine-tuning
        learning_rate: Learning rate (if None, uses LR from config)
        epochs: Number of epochs for fine-tuning
        
    Returns:
        model: Fine-tuned model
        avg_loss: Average loss during fine-tuning
    """
    if learning_rate is None:
        learning_rate = LR * 0.1  # Use a lower learning rate for fine-tuning
    
    # Configure optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Set model to training mode
    model.train()
    
    # Fine-tune for specified number of epochs
    total_loss = 0.0
    batch_count = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Move data to the same device as model
            device = next(model.parameters()).device
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Track statistics
            epoch_loss += loss.item()
            epoch_batches += 1
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Print epoch statistics
        if epoch_batches > 0:
            avg_epoch_loss = epoch_loss / epoch_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Calculate overall average loss
    avg_loss = total_loss / max(1, batch_count)
    
    return model, avg_loss

def adaptive_learning_rate(base_lr, round_idx, decay_factor=None, min_lr=None, decay_epochs=None):
    """
    Calculate adaptive learning rate based on round index and configuration.
    
    Args:
        base_lr: Base learning rate
        round_idx: Current round index
        decay_factor: Factor to decay learning rate (optional, uses LR_DECAY from config if None)
        min_lr: Minimum learning rate (optional, uses MIN_LR from config if None)
        decay_epochs: Apply decay every N epochs (optional, uses LR_DECAY_EPOCHS from config if None)
        
    Returns:
        current_lr: Current learning rate
    """
    # Use config values if parameters are not provided
    if decay_factor is None:
        decay_factor = LR_DECAY if 'LR_DECAY' in globals() else 0.95
    
    if min_lr is None:
        min_lr = MIN_LR if 'MIN_LR' in globals() else 0.0001
    
    if decay_epochs is None:
        decay_epochs = LR_DECAY_EPOCHS if 'LR_DECAY_EPOCHS' in globals() else 1
        
    # Calculate decay steps based on epoch interval
    if decay_epochs > 0 and round_idx > 0:
        decay_steps = round_idx // decay_epochs
        # Apply decay based on steps
        current_lr = base_lr * (decay_factor ** decay_steps)
    else:
        current_lr = base_lr
    
    # Ensure learning rate doesn't go below minimum
    current_lr = max(current_lr, min_lr)
    
    return current_lr

def extract_bn_parameters(model):
    """
    Extract BatchNorm parameters from a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        bn_params: Dictionary mapping parameter names to parameter values
    """
    bn_params = {}
    
    # Identify all BatchNorm layers and their parameters
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            # Store learnable parameters (weight and bias)
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    full_param_name = f"{name}.{param_name}" if name else param_name
                    bn_params[full_param_name] = param.clone().detach()
            
            # Also store running statistics (these are buffers, not parameters)
            for buffer_name, buffer in module.named_buffers():
                if 'running_mean' in buffer_name or 'running_var' in buffer_name or 'num_batches_tracked' in buffer_name:
                    full_buffer_name = f"{name}.{buffer_name}" if name else buffer_name
                    bn_params[full_buffer_name] = buffer.clone().detach()
    
    return bn_params

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed}")
    return seed

def model_to_vector(model):
    """
    Convert a model's parameters to a single flat vector.
    
    Args:
        model: PyTorch model
        
    Returns:
        Vector containing all model parameters
    """
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.data.view(-1))
    return torch.cat(params)

def get_gradient(model, data, target, loss_fn=None):
    """
    Compute the gradient of the model parameters with respect to the loss.
    
    Args:
        model: PyTorch model
        data: Input data batch
        target: Target labels batch
        loss_fn: Loss function (defaults to CrossEntropyLoss if None)
        
    Returns:
        Flattened gradient vector
    """
    # Use cross entropy loss if no loss function is specified
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Ensure data is on the correct device
    if data.device != device:
        data = data.to(device)
    if target.device != device:
        target = target.to(device)
    
    # Forward pass
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    
    # Backward pass
    loss.backward()
    
    # Collect gradients into a flat vector
    grad_vector = []
    for param in model.parameters():
        if param.requires_grad:
            if param.grad is None:
                # If no gradient, add zeros
                grad_vector.append(torch.zeros_like(param.data).view(-1))
            else:
                grad_vector.append(param.grad.data.view(-1))
    
    return torch.cat(grad_vector)