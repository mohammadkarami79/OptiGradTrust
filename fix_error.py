import re
import torch
import sys
import os

def fix_training_utils():
    print("Fixing CUDA assertion error in train_dual_attention function...")
    
    # Path to the training_utils.py file
    file_path = "federated_learning/training/training_utils.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the train_dual_attention function
    pattern = r'def train_dual_attention\(.*?(?=def|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Could not find train_dual_attention function in the file")
        return False
    
    # Replacement function
    new_function = '''def train_dual_attention(gradient_features, labels, global_context=None, 
                    epochs=100, batch_size=32, lr=0.001, weight_decay=1e-4,
                    device=None, verbose=True, early_stopping=10):
    """
    Train the enhanced dual attention model to detect malicious clients based on gradient features.
    
    Args:
        gradient_features (torch.Tensor): Feature vectors for each client gradient
        labels (torch.Tensor): Binary labels (1=honest, 0=malicious)
        global_context (torch.Tensor, optional): Global context vector
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        weight_decay (float): Weight decay for regularization
        device (torch.device): Device to train on (CPU/GPU)
        verbose (bool): Whether to print progress
        early_stopping (int): Patience for early stopping
        
    Returns:
        model (DualAttention): Trained dual attention model
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    from federated_learning.models.attention import DualAttention
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import sklearn.metrics as metrics
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors if they're not already
    if not isinstance(gradient_features, torch.Tensor):
        gradient_features = torch.tensor(gradient_features, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    # Move to device
    gradient_features = gradient_features.to(device)
    labels = labels.to(device)
    
    # Check for NaN values in features
    if torch.isnan(gradient_features).any():
        print("Warning: NaN values detected in gradient features, replacing with zeros")
        gradient_features = torch.nan_to_num(gradient_features, nan=0.0)
    
    if global_context is not None and not isinstance(global_context, torch.Tensor):
        global_context = torch.tensor(global_context, dtype=torch.float32).to(device)
        # Check for NaN values in global context
        if torch.isnan(global_context).any():
            print("Warning: NaN values detected in global context, replacing with zeros")
            global_context = torch.nan_to_num(global_context, nan=0.0)
    
    # Ensure labels are valid for BCE loss (between 0 and 1)
    labels = torch.clamp(labels, 0.0, 1.0)
    
    # Normalize features for better comparison
    feature_mean = gradient_features.mean(dim=0, keepdim=True)
    feature_std = gradient_features.std(dim=0, keepdim=True) + 1e-6
    normalized_features = (gradient_features - feature_mean) / feature_std
    
    # Check for NaN values after normalization
    if torch.isnan(normalized_features).any():
        print("Warning: NaN values detected after normalization, replacing with zeros")
        normalized_features = torch.nan_to_num(normalized_features, nan=0.0)
    
    # Compute statistics for honest clients' features
    honest_indices = (labels > 0.5).nonzero(as_tuple=True)[0]  # Labels: 1 for honest, 0 for malicious
    if len(honest_indices) > 0:
        honest_features = normalized_features[honest_indices]
        honest_mean = honest_features.mean(dim=0)
        honest_std = honest_features.std(dim=0)
        # Define thresholds for anomaly detection
        upper_thresholds = honest_mean + 2 * honest_std
        lower_thresholds = honest_mean - 2 * honest_std
    else:
        # If no honest clients, use global statistics
        upper_thresholds = torch.ones_like(feature_mean[0]) * 2.0
        lower_thresholds = torch.ones_like(feature_mean[0]) * -2.0
    
    # Sample weighting to focus on difficult examples
    sample_weights = torch.ones(len(gradient_features), device=device)
    abnormal_features = torch.zeros((len(gradient_features), gradient_features.shape[1]), device=device)
    detected_patterns = torch.zeros(len(gradient_features), device=device)
    
    # Identify specific patterns for malicious detection
    n_samples = len(gradient_features)
    n_features = gradient_features.shape[1]
    
    print(f"Analyzing {n_samples} clients with {n_features} features...")
    
    # Define common attack patterns
    patterns = [
        # Single feature abnormality (e.g., scaling attack)
        lambda feat: torch.any(torch.abs(feat) > 3.0),
        
        # High variance across features (sign-flipping)
        lambda feat: torch.std(feat) > 2.5,
        
        # Multiple features outside normal bounds
        lambda feat: torch.sum((feat > upper_thresholds) | (feat < lower_thresholds)) >= 2,
        
        # Opposite signs on correlated features
        lambda feat: any(feat[i] * feat[j] < -4.0 
                         for i in range(n_features) 
                         for j in range(i+1, n_features)),
        
        # Anomalous ratios between feature pairs
        lambda feat: any(torch.abs(feat[i]/feat[j]) > 5.0 if feat[j].abs() > 1e-3 else False
                         for i in range(n_features) 
                         for j in range(i+1, n_features))
    ]
    
    # Analyze each sample for patterns
    malicious_count = 0
    detected_by_pattern = [0] * len(patterns)
    
    for i in range(n_samples):
        is_malicious = labels[i].item() < 0.5  # 0 = malicious, 1 = honest
        feature_vector = normalized_features[i]
        
        # Check if sample matches any defined pattern
        for p_idx, pattern_fn in enumerate(patterns):
            try:
                if pattern_fn(feature_vector):
                    abnormal_features[i] += 1
                    detected_patterns[i] = 1
                    
                    # Increase weight for this sample if it's malicious
                    if is_malicious:
                        sample_weights[i] = 2.0  # Higher weight for malicious samples with known patterns
                        detected_by_pattern[p_idx] += 1
                        
                    break
            except Exception as e:
                # Skip failed pattern checks
                continue
        
        if is_malicious:
            malicious_count += 1
            
            # Feature-specific abnormality check
            for j in range(n_features):
                if feature_vector[j] > upper_thresholds[j] or feature_vector[j] < lower_thresholds[j]:
                    abnormal_features[i, j] = 1
                    
            # Check for abnormal feature pairs
            for j in range(n_features):
                for k in range(j+1, n_features):
                    # Check for unusual correlations or relationships
                    if (feature_vector[j] > upper_thresholds[j] and feature_vector[k] > upper_thresholds[k]) or \
                       (feature_vector[j] < lower_thresholds[j] and feature_vector[k] < lower_thresholds[k]):
                        # Both features are abnormal in same direction
                        abnormal_features[i, j] = 1
                        abnormal_features[i, k] = 1
                        sample_weights[i] = 2.5  # Higher weight for multi-feature patterns
    
    # Report detection results
    if malicious_count > 0:
        print(f"Found {malicious_count} malicious clients.")
        print(f"Pattern detection rate: {detected_patterns.sum().item() / malicious_count:.2f}")
        print("Detection by pattern type:")
        for i, count in enumerate(detected_by_pattern):
            print(f"  Pattern {i+1}: {count} clients")
    
    # Report weight distribution for malicious samples
    malicious_weights = sample_weights[labels < 0.5]  # 0 = malicious
    if len(malicious_weights) > 0:
        print(f"Malicious sample weights: min={malicious_weights.min().item():.2f}, "
              f"max={malicious_weights.max().item():.2f}, "
              f"mean={malicious_weights.mean().item():.2f}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(normalized_features, labels, sample_weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    feature_dim = gradient_features.shape[1]
    model = DualAttention(
        feature_dim=feature_dim,
        hidden_dim=64,
        num_heads=4,
        dropout=0.2
    )
    model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    criterion = nn.BCELoss(reduction='none')  # Use 'none' to apply sample weights
    
    # Training loop
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Separation loss coefficient (encourage separation between honest and malicious)
    separation_coef = 0.1
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        predictions = []
        confidences = []
        true_labels = []
        
        for batch_features, batch_labels, batch_weights in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            trust_scores, confidence_scores = model(batch_features, global_context)
            
            # Ensure values are valid for BCE loss (between 0 and 1)
            trust_scores = torch.clamp(trust_scores, 0.0 + 1e-7, 1.0 - 1e-7)
            confidence_scores = torch.clamp(confidence_scores, 0.0 + 1e-7, 1.0 - 1e-7) 
            batch_labels = torch.clamp(batch_labels, 0.0 + 1e-7, 1.0 - 1e-7)
            
            # Weighted BCE loss for trust scores
            bce_loss = criterion(trust_scores, batch_labels) 
            weighted_loss = (bce_loss * batch_weights).mean()
            
            # Confidence loss - encourage high confidence for correct predictions
            confidence_targets = (trust_scores.detach() > 0.5).float()
            confidence_targets = torch.clamp(confidence_targets, 0.0 + 1e-7, 1.0 - 1e-7)
            confidence_loss = F.binary_cross_entropy(confidence_scores, confidence_targets)
            
            # Only apply separation loss if we have both honest and malicious samples
            separation_loss = torch.tensor(0.0).to(device)
            honest_mask = batch_labels > 0.5
            malicious_mask = batch_labels < 0.5
            
            if honest_mask.any() and malicious_mask.any():
                honest_scores = trust_scores[honest_mask]
                malicious_scores = trust_scores[malicious_mask]
                
                # Calculate the mean scores for honest and malicious clients
                mean_honest = honest_scores.mean()
                mean_malicious = malicious_scores.mean()
                
                # Penalize if the gap between means is too small (honest should be higher)
                gap = mean_honest - mean_malicious
                separation_loss = torch.clamp(0.5 - gap, min=0.0)
            
            # Combined loss
            loss = weighted_loss + 0.1 * confidence_loss + separation_coef * separation_loss
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track statistics
            epoch_loss += loss.item()
            predictions.extend(trust_scores.detach().cpu().numpy())
            confidences.extend(confidence_scores.detach().cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())
        
        # Compute metrics
        epoch_loss /= len(dataloader)
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        true_labels = np.array(true_labels)
        binary_preds = (predictions > 0.5).astype(int)
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Calculate performance metrics
        try:
            accuracy = metrics.accuracy_score(true_labels, binary_preds)
            precision = metrics.precision_score(true_labels, binary_preds, zero_division=0)
            recall = metrics.recall_score(true_labels, binary_preds, zero_division=0)
            f1 = metrics.f1_score(true_labels, binary_preds, zero_division=0)
            
            # Calculate specificity (true negative rate)
            tn, fp, fn, tp = metrics.confusion_matrix(true_labels, binary_preds, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"\\nEpoch {epoch+1}/{epochs}:")
                print(f"Loss: {epoch_loss:.4f}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"Specificity: {specificity:.4f}")
                print(f"Mean Confidence: {confidences.mean():.4f}")
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def fix_server_file():
    print("Fixing issues in the Server class...")
    
    # Path to the server.py file
    file_path = "federated_learning/training/server.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the _pretrain_global_model method to handle CUDA device issues
    pattern = r'def _pretrain_global_model\(self\):.*?(?=def|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Could not find _pretrain_global_model method in the file")
        return False
    
    # Replacement method with fixes
    new_method = '''def _pretrain_global_model(self):
        """Pre-train the global model on the root dataset"""
        print("\\n=== Pretraining Global Model on Root Dataset ===")
        
        self.global_model.train()
        device = next(self.global_model.parameters()).device
        
        # Ensure data is moved to the correct device
        if self.root_loader is None:
            print("Warning: No root dataset available for pretraining")
            return
            
        # Configure optimizer
        optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
        
        # Train for specified number of epochs
        for epoch in range(LOCAL_EPOCHS_ROOT):
            running_loss = 0.0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.root_loader):
                # Move data to device and check for any issues
                try:
                    data = data.to(device)
                    target = target.to(device)
                except RuntimeError as e:
                    print(f"Error moving data to device: {str(e)}")
                    print(f"Data shape: {data.shape}, Target shape: {target.shape}")
                    print(f"Attempting to continue with CPU...")
                    device = torch.device('cpu')
                    self.global_model = self.global_model.to(device)
                    data = data.to(device)
                    target = target.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    outputs = self.global_model(data)
                    loss = F.cross_entropy(outputs, target)
                except RuntimeError as e:
                    print(f"Error in forward pass: {str(e)}")
                    print(f"Skipping batch {batch_idx}")
                    continue
                
                # Backward pass and optimize
                try:
                    loss.backward()
                    optimizer.step()
                except RuntimeError as e:
                    print(f"Error in backward pass: {str(e)}")
                    print(f"Skipping batch {batch_idx}")
                    continue
                
                # Track statistics
                running_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                if batch_idx % 50 == 0:
                    print(f"Pretrain Epoch {epoch+1}/{LOCAL_EPOCHS_ROOT}, "
                          f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Calculate average loss for this epoch
            if total_samples > 0:
                epoch_loss = running_loss / total_samples
                print(f"Pretrain Epoch {epoch+1}/{LOCAL_EPOCHS_ROOT} completed. "
                      f"Average loss: {epoch_loss:.4f}")
            else:
                print(f"Pretrain Epoch {epoch+1}/{LOCAL_EPOCHS_ROOT} completed without valid samples.")
        
        print("Pretraining completed")
'''
    
    # Replace the method in the content
    new_content = re.sub(pattern, new_method, content, flags=re.DOTALL)
    
    # Fix _generate_malicious_features method to handle NaN values
    pattern = r'def _generate_malicious_features\(self, honest_features\):.*?(?=def|\Z)'
    match = re.search(pattern, new_content, re.DOTALL)
    
    if match:
        # Replacement method with NaN handling
        new_method = '''def _generate_malicious_features(self, honest_features):
        """
        Generate synthetic malicious gradient features for training.
        
        Args:
            honest_features: Features from honest clients [batch_size, feature_dim]
            
        Returns:
            malicious_features: Synthetic malicious features [batch_size, feature_dim]
        """
        device = honest_features.device
        batch_size = honest_features.size(0)
        feature_dim = honest_features.size(1)
        
        # Verify feature dimension
        expected_dim = 6 if ENABLE_SHAPLEY else 5
        if feature_dim != expected_dim:
            print(f"Warning: Feature dimension mismatch. Expected {expected_dim}, got {feature_dim}")
        
        # Create synthetic malicious features with realistic attack patterns
        malicious_features = torch.zeros((batch_size, feature_dim), device=device)
        
        # Calculate statistics of honest features
        feature_means = honest_features.mean(dim=0)
        feature_stds = honest_features.std(dim=0) + 1e-6  # Avoid division by zero
        
        # Check for NaN values in statistics and replace them
        if torch.isnan(feature_means).any():
            print("Warning: NaN values detected in feature means, replacing with zeros")
            feature_means = torch.nan_to_num(feature_means, nan=0.0)
            
        if torch.isnan(feature_stds).any():
            print("Warning: NaN values detected in feature stds, replacing with ones")
            feature_stds = torch.nan_to_num(feature_stds, nan=1.0)
            
        # Common attack patterns
        attack_patterns = [
            # Pattern 1: Scaling Attack - extreme feature values 
            lambda: torch.normal(mean=0.0, std=2.0, size=(batch_size, feature_dim), device=device),
            
            # Pattern 2: High Reconstruction Error (first feature)
            lambda: torch.cat([
                torch.ones(batch_size, 1, device=device) * 0.9, 
                torch.normal(mean=feature_means[1:], std=feature_stds[1:] * 0.5, 
                            size=(batch_size, feature_dim-1), device=device)
            ], dim=1),
            
            # Pattern 3: Low similarity to root gradients (second feature)
            lambda: torch.cat([
                torch.normal(mean=feature_means[0], std=feature_stds[0] * 0.5, 
                             size=(batch_size, 1), device=device),
                torch.ones(batch_size, 1, device=device) * 0.2,  # Low root similarity
                torch.normal(mean=feature_means[2:], std=feature_stds[2:] * 0.5, 
                             size=(batch_size, feature_dim-2), device=device)
            ], dim=1),
            
            # Pattern 4: High magnitude (fourth feature)
            lambda: torch.cat([
                torch.normal(mean=feature_means[:3], std=feature_stds[:3] * 0.5, 
                             size=(batch_size, 3), device=device),
                torch.ones(batch_size, 1, device=device),  # High normalized norm
                torch.normal(mean=feature_means[4:], std=feature_stds[4:] * 0.5, 
                             size=(batch_size, feature_dim-4), device=device)
            ], dim=1)
        ]
        
        # Generate malicious features using random attack patterns
        for i in range(batch_size):
            # Select a random attack pattern
            pattern_idx = torch.randint(0, len(attack_patterns), (1,)).item()
            try:
                malicious_features[i] = attack_patterns[pattern_idx]()[i]
                
                # For Shapley values (if enabled), set to low values for malicious clients
                if feature_dim > 5:  # We have Shapley values
                    malicious_features[i, 5] = torch.rand(1, device=device).item() * 0.2  # Low Shapley value
            except Exception as e:
                print(f"Error generating malicious feature {i}: {str(e)}")
                # Fallback to a simple pattern
                malicious_features[i] = torch.cat([
                    torch.tensor([0.9, 0.2, 0.3, 1.0, 0.0], device=device),
                    torch.zeros(feature_dim - 5, device=device) + 0.1
                ])
        
        # Check for NaN values in the generated features
        if torch.isnan(malicious_features).any():
            print("Warning: NaN values detected in generated malicious features, replacing with zeros")
            malicious_features = torch.nan_to_num(malicious_features, nan=0.0)
            
        return malicious_features
'''
        # Replace the method in the content
        new_content = re.sub(pattern, new_method, new_content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    try:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Successfully updated {file_path}")
        return True
    except Exception as e:
        print(f"Error updating file: {str(e)}")
        return False

def fix_main_file():
    print("Fixing issues in the main.py file...")
    
    # Path to the main.py file
    file_path = "federated_learning/main.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the part where gradient features are being created for dual attention training
    pattern = r'# Train dual attention model using defined training function.*?try:.*?self\.dual_attention = train_dual_attention\((.*?)\)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # Extract the current arguments
        current_args = match.group(1)
        
        # Create the replacement with robust error handling
        replacement = '''# Train dual attention model using defined training function
                from federated_learning.training.training_utils import train_dual_attention
                try:
                    # Get feature dimension
                    feature_dim = X_train.shape[1]
                    
                    print(f"Training dual attention model with features shape: {X_train.shape}, labels shape: {y_train.shape}")
                    
                    # Check for issues with the data
                    if torch.isnan(X_train).any():
                        print("Warning: NaN values found in training features. Fixing...")
                        X_train = torch.nan_to_num(X_train, nan=0.0)
                    
                    # Create a new dual attention model with the correct feature dimension
                    from federated_learning.models.attention import DualAttention
                    dual_attention = DualAttention(
                        feature_dim=feature_dim,
                        hidden_dim=DUAL_ATTENTION_HIDDEN_SIZE,
                        num_heads=DUAL_ATTENTION_HEADS,
                        num_layers=DUAL_ATTENTION_LAYERS,
                        dropout=0.1
                    ).to(X_train.device)
                    
                    # Train the model with more robust error handling
                    self.dual_attention = train_dual_attention(
                        gradient_features=X_train,
                        labels=y_train,
                        epochs=50,
                        batch_size=min(16, len(X_train)),  # Smaller batch size for stability
                        lr=0.001
                    )'''
        
        # Replace in content
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Write the updated content back to the file
        try:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Successfully updated {file_path}")
            return True
        except Exception as e:
            print(f"Error updating file: {str(e)}")
            return False
    else:
        print("Could not find the train_dual_attention call in main.py")
        return False
    
def fix_vae_file():
    print("Fixing indentation issue in the VAE class...")
    
    # Path to the vae.py file
    file_path = "federated_learning/models/vae.py"
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the indentation issue in the else block
    content = content.replace(
        """        else:
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_dim),
            nn.Tanh()  # Normalize output
        )
            self.output_projection = None""",
        
        """        else:
            self.final_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], input_dim),
                nn.Tanh()  # Normalize output
            )
            self.output_projection = None"""
    )
    
    # Fix any other indentation issues in _init_weights method
    content = content.replace(
        """    def _init_weights(self):
        """Initialize weights using Xavier uniform for better convergence."""
        for module in self.modules():
        if isinstance(module, nn.Linear):""",
        
        """    def _init_weights(self):
        """Initialize weights using Xavier uniform for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):"""
    )
    
    # Write the updated content back to the file
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Successfully updated {file_path}")
        return True
    except Exception as e:
        print(f"Error updating file: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting to fix CUDA assertion errors in the codebase...")
    
    # Fix all files
    training_utils_fixed = fix_training_utils()
    server_fixed = fix_server_file()
    main_fixed = fix_main_file()
    fixed_vae = fix_vae_file()
    
    if training_utils_fixed and server_fixed and main_fixed and fixed_vae:
        print("\nAll fixes applied successfully! Try running the code again.")
    else:
        print("\nSome fixes could not be applied. Please check the error messages.") 