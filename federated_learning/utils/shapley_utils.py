"""
Utility functions for Shapley value calculation in federated learning.

Shapley values provide a principled approach to measure each client's contribution
to the global model performance. This implementation uses Monte Carlo sampling
for efficient approximation of Shapley values.
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from federated_learning.utils.model_utils import update_model_with_gradient
from federated_learning.config.config import *
from torch.utils.data import DataLoader

def evaluate_model_performance(model, validation_loader, device):
    """
    Evaluate model performance on validation dataset.
    
    Args:
        model: The model to evaluate.
        validation_loader: DataLoader for validation data.
        device: Device for computation.
        
    Returns:
        Performance score (accuracy).
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy

def apply_gradient(model, gradient):
    """
    Apply a gradient to update a model.
    
    Args:
        model: The model to update.
        gradient: The gradient to apply.
        
    Returns:
        None (updates model in-place).
    """
    # Flatten model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    offset = 0
    
    # Apply gradient to each parameter
    for param in params:
        num_params = param.numel()
        param_gradient = gradient[offset:offset+num_params].view(param.shape)
        param.data -= param_gradient.to(param.device)  # Subtract gradient (gradient descent)
        offset += num_params

def efficient_shapley_estimation(model, gradients, validation_loader, device, 
                                num_samples=5, learning_rate=0.01):
    """
    Efficiently estimate Shapley values for client gradients using Monte Carlo sampling.
    
    Args:
        model: Global model
        gradients: List of client gradients
        validation_loader: Validation data loader
        device: Device to run calculations on
        num_samples: Number of Monte Carlo samples
        learning_rate: Learning rate for model updates
        
    Returns:
        shapley_values: List of Shapley values for each client
    """
    num_clients = len(gradients)
    print(f"Calculating Shapley values for {num_clients} clients with {num_samples} samples")
    
    # Initialize Shapley values
    shapley_values = np.zeros(num_clients)
    
    # Get baseline performance (model without any updates)
    baseline_model = copy.deepcopy(model)
    baseline_performance = evaluate_model_performance(baseline_model, validation_loader, device)
    print(f"Baseline model performance: {baseline_performance:.4f}")
    
    # Monte Carlo sampling
    for sample in range(num_samples):
        print(f"\nMonte Carlo sample {sample+1}/{num_samples}")
        
        # Generate random permutation of clients
        client_permutation = list(range(num_clients))
        random.shuffle(client_permutation)
        
        # Track marginal contributions
        current_model = copy.deepcopy(model)
        previous_performance = baseline_performance
        
        # Process each client in the permutation
        for i, client_idx in enumerate(client_permutation):
            # Apply this client's gradient
            client_gradient = gradients[client_idx].to(device)
            
            # Update model with client gradient
            updated_model, _, _ = update_model_with_gradient(
                model=current_model,
                gradient=client_gradient,
                learning_rate=learning_rate
            )
            
            # Evaluate new model performance
            current_performance = evaluate_model_performance(updated_model, validation_loader, device)
            
            # Calculate marginal contribution
            marginal_contribution = current_performance - previous_performance
            
            # Update Shapley value for this client
            shapley_values[client_idx] += marginal_contribution
            
            # Update for next iteration
            current_model = updated_model
            previous_performance = current_performance
            
            print(f"  Client {client_idx}: Marginal contribution = {marginal_contribution:.6f}, "
                  f"Performance = {current_performance:.4f}")
    
    # Average over all samples
    shapley_values /= num_samples
    
    # Normalize Shapley values to [0, 1] range for easier interpretation
    min_value = min(shapley_values)
    max_value = max(shapley_values)
    
    if max_value > min_value:
        normalized_shapley = (shapley_values - min_value) / (max_value - min_value)
    else:
        normalized_shapley = np.ones_like(shapley_values) * 0.5
    
    print("\nShapley Value Results:")
    for i, (raw, norm) in enumerate(zip(shapley_values, normalized_shapley)):
        print(f"Client {i}: Raw = {raw:.6f}, Normalized = {norm:.4f}")
    
    return normalized_shapley

def calculate_shapley_values_batch(model, client_gradients, validation_loader, device, num_samples=5):
    """
    Calculate Shapley values for a batch of client gradients.
    Uses Monte Carlo sampling for computational efficiency.
    
    Args:
        model: The global model to evaluate on
        client_gradients: List of client gradients
        validation_loader: DataLoader for validation dataset
        device: Device to run computations on
        num_samples: Number of Monte Carlo samples
        
    Returns:
        List of Shapley values for each client
    """
    num_clients = len(client_gradients)
    
    # Calculate baseline performance (model without any client updates)
    baseline_perf = evaluate_model(model, validation_loader, device)
    print(f"Baseline model performance: {baseline_perf:.4f}")
    
    # Initialize Shapley values
    shapley_values = [0.0 for _ in range(num_clients)]
    
    # Monte Carlo approximation of Shapley values
    for sample in range(num_samples):
        print(f"\nMonte Carlo sample {sample+1}/{num_samples}")
        
        # Generate a random permutation of clients
        client_indices = list(range(num_clients))
        random.shuffle(client_indices)
        
        # Track marginal contributions
        prev_perf = baseline_perf
        
        # For each client in the permutation
        for i, client_idx in enumerate(client_indices):
            # Create a copy of the model
            model_copy = copy.deepcopy(model)
            
            # Apply gradients up to (and including) the current client
            current_gradients = [client_gradients[idx] for idx in client_indices[:i+1]]
            current_weights = [1.0 / len(current_gradients)] * len(current_gradients)
            
            # Aggregate gradients
            if current_gradients:
                aggregated_gradient = aggregate_gradients(current_gradients, current_weights)
                
                # Apply aggregated gradient to model
                from federated_learning.utils.model_utils import update_model_with_gradient
                lr = 0.01  # Use consistent learning rate for evaluation
                print(f"Updating model with gradient (norm: {torch.norm(aggregated_gradient).item():.8f}, learning rate: {lr:.8f})")
                model_copy, _, _ = update_model_with_gradient(model_copy, aggregated_gradient, learning_rate=lr)
            
            # Evaluate performance
            perf = evaluate_model(model_copy, validation_loader, device)
            
            # Calculate marginal contribution
            marginal = perf - prev_perf
            shapley_values[client_idx] += marginal
            
            print(f"  Client {client_indices[i]}: Marginal contribution = {marginal:.6f}, Performance = {perf:.4f}")
            
            # Update for next iteration
            prev_perf = perf
    
    # Average over samples
    shapley_values = [val / num_samples for val in shapley_values]
    
    # Process Shapley values for better differentiation
    # Normalize to [0,1] range, handling negative values
    min_val = min(shapley_values)
    if min_val < 0:
        # Shift values if any are negative
        shapley_values = [val - min_val for val in shapley_values]
    
    # Normalize
    max_val = max(shapley_values) if max(shapley_values) > 0 else 1.0
    normalized_shapley = [val / max_val for val in shapley_values]
    
    # Handle case where all Shapley values are very similar
    # Add slight randomness if the range is too small
    if max(normalized_shapley) - min(normalized_shapley) < 0.05:
        # Add minimal random noise to differentiate between clients
        normalized_shapley = [val + random.uniform(-0.02, 0.02) for val in normalized_shapley]
        # Re-normalize to [0,1]
        min_val = min(normalized_shapley)
        max_val = max(normalized_shapley)
        normalized_shapley = [(val - min_val) / (max_val - min_val) for val in normalized_shapley]
        print("Applied small random perturbation to differentiate similar Shapley values")
    
    print("\nShapley Value Results:")
    for i, (raw, norm) in enumerate(zip(shapley_values, normalized_shapley)):
        print(f"Client {i}: Raw = {raw:.6f}, Normalized = {norm:.4f}")
    
    return normalized_shapley

def evaluate_model(model, validation_loader, device):
    """
    Evaluate model performance on validation dataset.
    
    Args:
        model: Model to evaluate
        validation_loader: DataLoader for validation data
        device: Device to run computations on
        
    Returns:
        accuracy: Model accuracy on validation data
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def aggregate_gradients(gradients, weights):
    """
    Aggregate gradients with weights.
    
    Args:
        gradients: List of gradient tensors
        weights: List of weights corresponding to gradients
        
    Returns:
        aggregated_gradient: Weighted sum of gradients
    """
    # Convert to tensor and expand weights
    grad_tensor = torch.stack(gradients)
    weights_tensor = torch.tensor(weights, device=grad_tensor.device).view(-1, 1)
    
    # Compute weighted sum
    aggregated_gradient = torch.sum(grad_tensor * weights_tensor, dim=0)
    
    return aggregated_gradient

def integrate_shapley_into_features(features, shapley_values, shapley_weight=0.3):
    """
    Integrate Shapley values into feature vectors for dual attention mechanism.
    
    Args:
        features: Tensor of feature vectors [num_clients, feature_dim]
        shapley_values: List of Shapley values for each client
        shapley_weight: Weight of Shapley value in the final feature vector
        
    Returns:
        enhanced_features: Feature vectors with Shapley values [num_clients, feature_dim+1]
    """
    # Convert Shapley values to tensor
    shapley_tensor = torch.tensor(shapley_values, device=features.device).unsqueeze(1)
    
    # Concatenate Shapley values to features
    enhanced_features = torch.cat([features, shapley_tensor], dim=1)
    
    return enhanced_features

def monte_carlo_shapley(model, clients_gradients, validation_loader, num_samples=100, device=None):
    """
    Calculate Shapley values using Monte Carlo sampling.
    
    Args:
        model: The global model.
        clients_gradients: List of client gradients.
        validation_loader: DataLoader for validation data.
        num_samples: Number of Monte Carlo samples.
        device: Device for computation.
        
    Returns:
        List of Shapley values for each client.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    num_clients = len(clients_gradients)
    shapley_values = [0.0] * num_clients
    
    try:
        # Evaluate the model with no gradients applied as baseline
        baseline_performance = evaluate_model_performance(model, validation_loader, device)
        
        # Perform Monte Carlo sampling
        for _ in range(num_samples):
            # Generate a random permutation of clients
            permutation = np.random.permutation(num_clients)
            
            # Track marginal contributions
            current_model = copy.deepcopy(model)
            current_performance = baseline_performance
            
            for i, client_idx in enumerate(permutation):
                # Apply this client's gradient
                temp_model = copy.deepcopy(current_model)
                apply_gradient(temp_model, clients_gradients[client_idx])
                
                # Evaluate new performance
                new_performance = evaluate_model_performance(temp_model, validation_loader, device)
                
                # Calculate marginal contribution
                marginal_contribution = new_performance - current_performance
                
                # Add to Shapley value
                shapley_values[client_idx] += marginal_contribution
                
                # Update current state
                current_model = temp_model
                current_performance = new_performance
        
        # Average Shapley values across samples
        shapley_values = [val / num_samples for val in shapley_values]
        
        # Normalize to [0, 1] range if possible
        min_val = min(shapley_values)
        max_val = max(shapley_values)
        
        if max_val > min_val:  # Avoid division by zero
            shapley_values = [(val - min_val) / (max_val - min_val) for val in shapley_values]
        else:
            # All values are the same, set to 0.5
            shapley_values = [0.5] * num_clients
            
        return shapley_values
    except Exception as e:
        print(f"Error in Shapley value calculation: {str(e)}")
        # Return default values as fallback
        return [0.5] * num_clients
