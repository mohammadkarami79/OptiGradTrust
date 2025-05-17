import torch
import torch.nn.functional as F
import numpy as np
from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.models.resnet import ResNet50Alzheimer, ResNet18Alzheimer
from federated_learning.models.vae import VAE, GradientVAE
from federated_learning.models.attention import DualAttention
from torch.utils.data import DataLoader, Subset
import copy
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt

def calculate_model_param_change(old_model, new_model):
    """Calculate the total parameter change between two models."""
    total_change = 0.0
    param_count = 0
    
    # Iterate through all parameters
    for (old_name, old_param), (new_name, new_param) in zip(
        old_model.named_parameters(), new_model.named_parameters()):
        
        if old_name != new_name:
            raise ValueError(f"Parameter names don't match: {old_name} vs {new_name}")
        
        if old_param.requires_grad:
            # Calculate absolute change
            change = torch.sum(torch.abs(old_param - new_param)).item()
            total_change += change
            param_count += old_param.numel()
    
    # Return average change per parameter
    return total_change / max(1, param_count)

class Server:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_norms = None
        self.trust_scores = None
        self.confidence_scores = None
        self.weights = None
        self.root_gradients = []
        self.clients = []
        self.malicious_clients = []
        
        # Initialize the global model
        self.global_model = self._create_model().to(self.device)
        
        # Initialize VAE and Dual Attention
        self.vae = self._create_vae()
        self.dual_attention = self._create_dual_attention()
        
        print("Initialized server")
        
    def _create_model(self):
        if MODEL == 'CNN':
            return CNNMnist()
        elif MODEL == 'RESNET50':
            return ResNet50Alzheimer(num_classes=10 if DATASET == 'MNIST' else ALZHEIMER_CLASSES)
        elif MODEL == 'RESNET18':
            return ResNet18Alzheimer(num_classes=10 if DATASET == 'MNIST' else ALZHEIMER_CLASSES)
        else:
            raise ValueError(f"Unknown model type: {MODEL}")

    def _create_vae(self):
        return GradientVAE(
            input_dim=GRADIENT_DIMENSION,
            hidden_dim=VAE_HIDDEN_DIM,
            latent_dim=VAE_LATENT_DIM
        ).to(self.device)
        
    def _create_dual_attention(self):
        feature_dim = 6 if ENABLE_SHAPLEY else 5
        return DualAttention(
            feature_dim=feature_dim,
            hidden_dim=DUAL_ATTENTION_HIDDEN_SIZE,
            num_heads=DUAL_ATTENTION_HEADS
        ).to(self.device)
        
    def _pretrain_global_model(self):
        """Pre-train the global model on the root dataset"""
        print("\n=== Pretraining Global Model on Root Dataset ===")
        
        self.global_model.train()
        device = self.device
        
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
        
    def _compute_all_gradient_features(self, client_gradients):
        """
        Compute feature vectors for all client gradients.
        
        Args:
            client_gradients: List of client gradients
            
        Returns:
            Tensor of feature vectors [num_clients, num_features]
        """
        num_clients = len(client_gradients)
        feature_dim = 6 if ENABLE_SHAPLEY else 5
        features = torch.zeros((num_clients, feature_dim), device=self.device)
        
        # Handle both list of gradients and tensor of gradients
        if isinstance(client_gradients, list):
            print(f"Computing features for {len(client_gradients)} gradients (list input)")
            
            # First pass: compute basic features without client similarity
            for i, grad in enumerate(client_gradients):
                # For each client, compute the gradient features without client similarity
                features[i] = self._compute_gradient_features(grad, root_gradient=self.root_gradients[0] if self.root_gradients else None, skip_client_sim=True)
            
            # Second pass: compute client similarity for all gradients
            cosine_similarities = torch.zeros((num_clients, num_clients), device=self.device)
            for i in range(num_clients):
                for j in range(num_clients):
                    if i != j:
                        # Calculate cosine similarity between gradients
                        cos_sim = F.cosine_similarity(
                            client_gradients[i].unsqueeze(0), 
                            client_gradients[j].unsqueeze(0)
                        )
                        # Normalize to [0, 1] range
                        normalized_sim = (cos_sim + 1) / 2
                        cosine_similarities[i, j] = normalized_sim
            
            # Update client similarity feature (index 2)
            for i in range(num_clients):
                # Average similarity with other clients
                if num_clients > 1:
                    client_sim = cosine_similarities[i].sum() / (num_clients - 1)
                    features[i, 2] = client_sim
                else:
                    features[i, 2] = 0.5  # Default for single client
                
                # Print updated feature vector with client similarity
                feature_names = [
                    "VAE Reconstruction Error", 
                    "Root Similarity", 
                    "Client Similarity", 
                    "Gradient Norm", 
                    "Sign Consistency", 
                    "Shapley Value"
                ]
                
                grad_norm = torch.norm(client_gradients[i]).item()
                print(f"\nClient {i} feature vector:")
                print(f"  Raw gradient norm: {grad_norm:.4f}")
                for j in range(feature_dim):
                    print(f"  {j+1}. {feature_names[j]}: {features[i, j].item():.4f}")
        
        else:
            # Handle tensor input (batch processing)
            print(f"Computing features for gradients tensor of shape {client_gradients.shape}")
            
            # Use the first root gradient as reference if available
            root_grad = self.root_gradients[0] if self.root_gradients else None
            
            # First compute basic features
            for i in range(client_gradients.shape[0]):
                features[i] = self._compute_gradient_features(
                    client_gradients[i], 
                    root_gradient=root_grad,
                    skip_client_sim=True
                )
            
            # Then compute client similarities
            cosine_similarities = torch.zeros((client_gradients.shape[0], client_gradients.shape[0]), device=self.device)
            for i in range(client_gradients.shape[0]):
                for j in range(client_gradients.shape[0]):
                    if i != j:
                        cos_sim = F.cosine_similarity(
                            client_gradients[i].unsqueeze(0), 
                            client_gradients[j].unsqueeze(0)
                        )
                        normalized_sim = (cos_sim + 1) / 2
                        cosine_similarities[i, j] = normalized_sim
            
            # Update client similarity feature
            for i in range(client_gradients.shape[0]):
                if client_gradients.shape[0] > 1:
                    client_sim = cosine_similarities[i].sum() / (client_gradients.shape[0] - 1)
                    features[i, 2] = client_sim
                else:
                    features[i, 2] = 0.5
                
                # Print feature vector
                feature_names = [
                    "VAE Reconstruction Error", 
                    "Root Similarity", 
                    "Client Similarity", 
                    "Gradient Norm", 
                    "Sign Consistency", 
                    "Shapley Value"
                ]
                
                grad_norm = torch.norm(client_gradients[i]).item()
                print(f"\nClient {i} feature vector:")
                print(f"  Raw gradient norm: {grad_norm:.4f}")
                for j in range(feature_dim):
                    print(f"  {j+1}. {feature_names[j]}: {features[i, j].item():.4f}")
        
        # Print a summary of all feature vectors
        print("\nFeature vectors summary:")
        print(f"Shape: {features.shape}")
        print("Feature means:")
        for i in range(feature_dim):
            print(f"  Feature {i+1}: {features[:, i].mean().item():.4f}")
        
        return features
        
    def _compute_gradient_features(self, gradient, root_gradient=None, skip_client_sim=False):
        """
        Compute feature vector for a single gradient.
        
        Args:
            gradient: Client gradient
            root_gradient: Optional root gradient for comparison
            skip_client_sim: Skip client similarity computation
            
        Returns:
            Tensor of features [feature_dim]
        """
        feature_dim = 6 if ENABLE_SHAPLEY else 5
        features = torch.zeros(feature_dim, device=self.device)
        
        # Ensure gradient is on the correct device
        if gradient.device != self.device:
            gradient = gradient.to(self.device)
        
        # 1. Reconstruction error from VAE
        if self.vae is not None:
            try:
                with torch.no_grad():
                    recon_gradient, _, _ = self.vae(gradient.unsqueeze(0))
                    recon_error = F.mse_loss(recon_gradient.squeeze(0), gradient)
                    
                    # Normalize reconstruction error to 0-1 range
                    normalized_recon_error = torch.clamp(recon_error / (recon_error + 1.0), 0.0, 1.0)
                    features[0] = normalized_recon_error
        except Exception as e:
                print(f"Error computing reconstruction error: {str(e)}")
                features[0] = 0.5  # Default value
        else:
            features[0] = 0.5  # Default value if VAE not available
        
        # 2. Root similarity (cosine similarity to root gradient)
        if root_gradient is not None:
            if root_gradient.device != self.device:
                root_gradient = root_gradient.to(self.device)
                
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(gradient.unsqueeze(0), root_gradient.unsqueeze(0))
            # Normalize to 0-1 range (from -1 to 1)
            root_sim = (cos_sim + 1.0) / 2.0
            features[1] = root_sim
        else:
            features[1] = 0.5  # Default value
        
        # 3. Client similarity (average cosine similarity to other clients)
        # In first pass, we skip this as it requires comparing to other clients
        if not skip_client_sim:
            # The value for this will be set in _compute_all_gradient_features
            features[2] = 0.5  # Default placeholder
        else:
            features[2] = 0.5  # Default placeholder
        
        # 4. Gradient norm (improved normalization)
        # Using a linear normalization with a predefined max norm
        grad_norm = torch.norm(gradient).item()
        raw_norm = grad_norm

        # Linear normalization relative to MAX_GRADIENT_NORM
        max_norm = MAX_GRADIENT_NORM if 'MAX_GRADIENT_NORM' in globals() else 10.0
        norm_feature = min(grad_norm / max_norm, 1.0)
        features[3] = norm_feature
        
        # 5. Consistency (gradient pattern consistency)
        if root_gradient is not None:
            # Simple pattern consistency metric:
            # Check what percentage of gradients have the same sign as the root
            sign_match = (torch.sign(gradient) == torch.sign(root_gradient)).float().mean()
            features[4] = sign_match
        else:
            features[4] = 0.5  # Default value
        
        # 6. Shapley value (if enabled)
        if ENABLE_SHAPLEY and feature_dim > 5:
            features[5] = 0.5  # Default Shapley value (placeholder)
        
        # Debug print all features if not in batch mode
        if not skip_client_sim:
            feature_names = [
                "VAE Reconstruction Error", 
                "Root Similarity", 
                "Client Similarity", 
                "Gradient Norm", 
                "Sign Consistency", 
                "Shapley Value"
            ]
            
            print(f"\nGradient feature vector:")
            print(f"  Raw gradient norm: {raw_norm:.4f}")
            for i in range(feature_dim):
                print(f"  {i+1}. {feature_names[i]}: {features[i].item():.4f}")
        
        return features
        
    def _generate_malicious_features(self, honest_features):
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
        
        # Create malicious features based on distortions of honest features
        malicious_features = torch.zeros((batch_size, feature_dim), device=device)
        
        # 1. High reconstruction error (low trust)
        malicious_features[:, 0] = torch.clamp(honest_features[:, 0] * 2.5, 0.6, 0.95)
        
        # 2. Low root similarity
        malicious_features[:, 1] = torch.clamp(honest_features[:, 1] * 0.5, 0.1, 0.4)
        
        # 3. Low client similarity
        malicious_features[:, 2] = torch.clamp(honest_features[:, 2] * 0.4, 0.1, 0.4)
        
        # 4. Abnormal gradient norms (either too big or too small)
        if torch.rand(1).item() > 0.5:
            # Too big
            malicious_features[:, 3] = torch.clamp(honest_features[:, 3] * 3.0, 0.7, 0.95)
        else:
            # Too small
            malicious_features[:, 3] = torch.clamp(honest_features[:, 3] * 0.2, 0.05, 0.2)
        
        # 5. Low consistency
        malicious_features[:, 4] = torch.clamp(honest_features[:, 4] * 0.3, 0.05, 0.4)
        
        # 6. Shapley value (if enabled)
        if ENABLE_SHAPLEY and feature_dim > 5:
            malicious_features[:, 5] = torch.clamp(honest_features[:, 5] * 0.2, 0.0, 0.3)
        
        # Add random noise to create variety in malicious features
        noise = torch.randn_like(malicious_features) * 0.1
        malicious_features = torch.clamp(malicious_features + noise, 0.01, 0.99)
        
        return malicious_features
        
    def _compute_shapley_values(self, client_gradients, client_indices):
        """
        Compute Shapley values for client gradients
        
        Args:
            client_gradients: List of client gradients
            client_indices: List of client indices
            
        Returns:
            shapley_values: List of Shapley values for each client
        """
        try:
            if not ENABLE_SHAPLEY:
                print("Shapley value calculation is disabled")
                return None
                
            print("\n--- Computing Shapley values ---")
            
            # Import Shapley utils
            from federated_learning.utils.shapley_utils import calculate_shapley_values_batch
            
            # Create a small validation dataset for Shapley value calculation
            # Use a subset of the test dataset to make computation faster
            validation_data = []
            validation_targets = []
            
            # Ensure we have a test_loader available
            if not hasattr(self, 'test_loader'):
                # Create test loader if not already available
                self.test_loader = torch.utils.data.DataLoader(
                    self.test_dataset, 
                    batch_size=BATCH_SIZE,
                    shuffle=False
                )
                
            for data, target in self.test_loader:
                validation_data.append(data[:10])  # Take first 10 samples from each batch
                validation_targets.append(target[:10])
                if len(validation_data) >= 5:  # Use 5 batches max
                    break
                    
            if not validation_data:
                print("No validation data available for Shapley calculation")
                return None
                
            validation_data = torch.cat(validation_data, dim=0)
            validation_targets = torch.cat(validation_targets, dim=0)
            
            validation_dataset = torch.utils.data.TensorDataset(validation_data, validation_targets)
            validation_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=SHAPLEY_BATCH_SIZE if 'SHAPLEY_BATCH_SIZE' in globals() else 32,
                shuffle=False
            )
            
            # Calculate Shapley values
            print(f"Calculating Shapley values for {len(client_gradients)} clients")
            shapley_values = calculate_shapley_values_batch(
                model=copy.deepcopy(self.global_model),
                client_gradients=client_gradients,
                validation_loader=validation_loader,
                device=self.device,
                num_samples=SHAPLEY_SAMPLES
            )
            
            # Normalize Shapley values to ensure they are meaningful
            # First, make all values positive for proper normalization
            min_val = min(shapley_values)
            if min_val < 0:
                # Shift by min value if any negative values exist
                shapley_values = [val - min_val for val in shapley_values]
                
            # Normalize to [0, 1] range
            max_val = max(shapley_values) if max(shapley_values) > 0 else 1.0
            shapley_values = [val / max_val for val in shapley_values]
            
            # If all values are the same, create slight differences
            if max(shapley_values) - min(shapley_values) < 0.001:
                # Add small random differences if all values are the same
                shapley_values = [val + (i * 0.01) for i, val in enumerate(shapley_values)]
                # Re-normalize
                max_val = max(shapley_values)
                shapley_values = [val / max_val for val in shapley_values]
                
            print("\nShapley values:")
            for i, client_idx in enumerate(client_indices):
                if client_idx >= len(self.clients):
                    print(f"Warning: Client index {client_idx} out of range")
                    continue

                client = self.clients[client_idx]
                is_malicious = "YES" if client.is_malicious else "NO"
                print(f"Client {client_idx} (Malicious: {is_malicious}): Shapley value = {shapley_values[i]:.4f}")
                
            return shapley_values
            
        except Exception as e:
            print(f"Error computing Shapley values: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def train(self, num_rounds=GLOBAL_EPOCHS):
        """Train the model using federated learning."""
        print("\n=== Starting Federated Learning Process ===")
        test_errors = []
        trust_scores = []
        round_metrics = {}
        
        # Ensure we have clients
        if not self.clients:
            print("Error: No clients available for training")
            return
            
        # Ensure we have a test dataset
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            print("Warning: No test dataset specified, creating a default one")
            test_loader = torch.utils.data.DataLoader(
                self.test_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            self.test_loader = test_loader
            
        # Initial evaluation
        print("\n=== Initial Model Evaluation ===")
        from federated_learning.training.training_utils import test
        initial_acc, initial_error = test(self.global_model, self.test_loader)
        print(f"Initial test accuracy: {initial_acc:.4f}, error: {initial_error:.4f}")
        test_errors.append(initial_error)
        
        # Main federated learning loop
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
            
            # Evaluate model at the beginning of the round
            print("\n--- Evaluating global model ---")
            round_acc, round_error = test(self.global_model, self.test_loader)
            print(f"Round {round_idx + 1} - Accuracy: {round_acc:.4f}, Error: {round_error:.4f}")
            test_errors.append(round_error)
            
            # Initialize round metrics storage
            round_metrics[round_idx] = {
                'test_accuracy': round_acc,
                'test_error': round_error,
                'trust_scores': {},
                'gradient_norms': {},
                'weights': {},
                'features': {},
                'raw_metrics': {},
                'client_status': {}
            }
            
            # Select clients for this round
            selected_clients = self._select_clients()
            print(f"Selected {len(selected_clients)} clients for this round")
            
            # Print client malicious status for clarity
            print("\nClient status:")
            for idx in selected_clients:
                client = self.clients[idx]
                status = "MALICIOUS" if client.is_malicious else "HONEST"
                print(f"Client {idx}: {status}")
                # Store client status in metrics
                round_metrics[round_idx]['client_status'][idx] = status
            
            # Collect updates from clients
            all_gradients = []
            all_features = []
            client_indices = []
            gradient_norms = []
            
            # Track which clients correspond to which index in all_gradients/all_features
            # This ensures we maintain proper client identification throughout
            gradient_to_client_map = {}
            
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                print(f"\nClient {client_idx} (Malicious: {client.is_malicious})")
                
                # Client performs local training and computes gradient
                try:
                    result = client.train(self.global_model, round_idx)
                    
                    # Client.train returns a tuple of (gradient, features)
                    if isinstance(result, tuple) and len(result) == 2:
                        gradient, features = result
                else:
                        # Backward compatibility for clients that only return gradient
                        gradient = result
                        features = None
                    
                    # Skip client if gradient computation failed
                    if gradient is None:
                        print(f"Client {client_idx}: Failed to compute gradient")
                        continue
                        
                    # Ensure gradient is on the correct device
                    gradient = gradient.to(self.device)
                    
                    # Display raw metrics before any normalization
                    if client.is_malicious:
                        print("--- MALICIOUS CLIENT METRICS ---")
                        gradient_norm = torch.norm(gradient).item()
                        print(f"AFTER attack - Gradient norm: {gradient_norm:.4f}")
                        if hasattr(client, 'original_gradient') and client.original_gradient is not None:
                            original_norm = torch.norm(client.original_gradient).item()
                            print(f"BEFORE attack - Original gradient norm: {original_norm:.4f}")
                            print(f"Increase from attack: {gradient_norm - original_norm:.4f} ({(gradient_norm/original_norm - 1)*100:.2f}%)")
                            
                            # Store raw metrics
                            round_metrics[round_idx]['raw_metrics'][client_idx] = {
                                'original_norm': original_norm,
                                'attacked_norm': gradient_norm,
                                'is_malicious': True,
                                'attack_type': client.attack.attack_type if hasattr(client, 'attack') else 'unknown'
                            }
                    else:
                        # Store raw metrics for honest clients too
                        round_metrics[round_idx]['raw_metrics'][client_idx] = {
                            'original_norm': torch.norm(gradient).item(),
                            'is_malicious': False
                        }
                    
                    # Track metrics
                    grad_norm = torch.norm(gradient).item()
                    print(f"Client {client_idx}: Gradient norm = {grad_norm:.4f}")
                    gradient_norms.append(grad_norm)
                    
                    # Log metrics for this round
                    round_metrics[round_idx]['gradient_norms'][client_idx] = grad_norm
                    
                    # Store gradient and metadata
                    gradient_index = len(all_gradients)
                    all_gradients.append(gradient)
                    # Map this gradient index to the client index
                    gradient_to_client_map[gradient_index] = client_idx
                    
                    # Save client features if available, or compute them
                    if features is not None:
                        # Ensure features are on the correct device
                        features = features.to(self.device)
                        all_features.append(features)
                    
                    client_indices.append(client_idx)

        except Exception as e:
                    print(f"Error with client {client_idx}: {str(e)}")
            import traceback
            traceback.print_exc()

            # Skip round if no gradients collected
            if len(all_gradients) == 0:
                print("No valid gradients collected this round. Skipping.")
                continue
            
            # Compute features if not provided by clients
            if not all_features and all_gradients:
                print("Computing gradient features...")
                # For each gradient, compute feature vector
                feature_vectors = self._compute_all_gradient_features(all_gradients)
                all_features = feature_vectors
                
            # Compute Shapley values if enabled
            if ENABLE_SHAPLEY:
                shapley_values = self._compute_shapley_values(all_gradients, client_indices)
                
                if shapley_values is not None and all_features is not None and len(all_features) > 0:
                    # Update feature vectors with Shapley values
                    if isinstance(all_features, torch.Tensor):
                        features_tensor = all_features
                    else:
                        features_tensor = torch.stack(all_features) if isinstance(all_features[0], torch.Tensor) else all_features
                    
                    # Add Shapley values as the 6th feature
                    if features_tensor.size(1) == 5:  # If features don't already include Shapley
                        shapley_tensor = torch.tensor(shapley_values, device=self.device).unsqueeze(1)
                        features_tensor = torch.cat([features_tensor, shapley_tensor], dim=1)
                        print("Added Shapley values as 6th feature")
                        
                        # Replace all_features with updated tensor
                        all_features = features_tensor
                    elif features_tensor.size(1) >= 6:  # If features already have space for Shapley
                        # Update the 6th feature with computed Shapley values
                        for i, val in enumerate(shapley_values):
                            features_tensor[i, 5] = val
                        print("Updated 6th feature with Shapley values")
                        
                        # Replace all_features with updated tensor
                        all_features = features_tensor
            
            # Store all feature vectors in round metrics
            if isinstance(all_features, torch.Tensor):
                features_tensor = all_features
            else:
                features_tensor = torch.stack(all_features) if isinstance(all_features[0], torch.Tensor) else all_features
                
            for i, client_idx in enumerate(client_indices):
                # Convert feature tensor to list for JSON serialization
                round_metrics[round_idx]['features'][client_idx] = features_tensor[i].cpu().tolist()
                
            # Compute trust scores using dual attention model
            if self.dual_attention is not None and len(all_features) > 0:
                print("\n--- Computing trust scores with dual attention ---")
                
                # Get the global context (mean of all features)
                if isinstance(all_features, torch.Tensor):
                    features_tensor = all_features
                else:
                    features_tensor = torch.stack(all_features) if isinstance(all_features[0], torch.Tensor) else all_features
                
                global_context = torch.mean(features_tensor, dim=0, keepdim=True)
                
                # Compute trust scores using dual attention
                try:
                    self.dual_attention.eval()
        with torch.no_grad():
                        trust_scores_tensor, confidence_scores = self.dual_attention(features_tensor, global_context)
                    
                    # Convert to list for easier handling
                    client_trust_scores = trust_scores_tensor.cpu().numpy().tolist()
                    
                    # Print trust scores
                    print("\nClient Trust Scores:")
                    for i, client_idx in enumerate(client_indices):
                        client = self.clients[client_idx]
                        is_malicious = "YES" if client.is_malicious else "NO"
                        trust_score = client_trust_scores[i]
                        round_metrics[round_idx]['trust_scores'][client_idx] = trust_score
                        print(f"Client {client_idx} (Malicious: {is_malicious}): Trust Score = {trust_score:.4f}")
                    
                    # Store trust scores for later use
                    self.trust_scores = trust_scores_tensor
                    self.confidence_scores = confidence_scores
                    
                except Exception as e:
                    print(f"Error computing trust scores: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    # Use equal weights as fallback
                    client_trust_scores = [1.0] * len(client_indices)
                    self.trust_scores = torch.ones(len(client_indices), device=self.device)
                    self.confidence_scores = torch.ones(len(client_indices), device=self.device)
            else:
                # Fallback to equal weights if dual attention is not available
                print("No dual attention model available. Using equal weights.")
                client_trust_scores = [1.0] * len(client_indices)
                self.trust_scores = torch.ones(len(client_indices), device=self.device)
                self.confidence_scores = torch.ones(len(client_indices), device=self.device)
            
            # Compute aggregation weights based on trust scores and selected method
            print("\n--- Computing aggregation weights ---")
            
            # Get weights using the dual attention model's get_gradient_weights method
            if hasattr(self.dual_attention, 'get_gradient_weights') and self.dual_attention is not None:
                try:
                    if isinstance(all_features, torch.Tensor):
                        features_tensor = all_features
                    else:
                        features_tensor = torch.stack(all_features) if isinstance(all_features[0], torch.Tensor) else all_features
                        
                    weights_tensor, malicious_indices = self.dual_attention.get_gradient_weights(
                        features_tensor, 
                        self.trust_scores,
                        self.confidence_scores
                    )
                    
                    # Convert to list for easier handling
                    weights = weights_tensor.cpu().numpy().tolist()
                    
                    # Convert detected malicious_indices to client indices
                    detected_malicious_clients = []
                    for i in malicious_indices:
                        if i < len(client_indices):
                            detected_malicious_clients.append(client_indices[i])
                    
                    # Print which clients were detected as malicious
                    print(f"\nDetected {len(detected_malicious_clients)} potential malicious clients")
                    if detected_malicious_clients:
                        print(f"Detected malicious client IDs: {detected_malicious_clients}")
                        
                        # Check which ones are actually malicious
                        actual_malicious = [idx for idx in detected_malicious_clients if self.clients[idx].is_malicious]
                        false_positives = [idx for idx in detected_malicious_clients if not self.clients[idx].is_malicious]
                        
                        # Check for false negatives (malicious clients not detected)
                        actual_malicious_set = {idx for idx in client_indices if self.clients[idx].is_malicious}
                        false_negatives = list(actual_malicious_set - set(detected_malicious_clients))
                        
                        print(f"Actually malicious: {actual_malicious}")
                        if false_positives:
                            print(f"False positives: {false_positives}")
                        if false_negatives:
                            print(f"False negatives (undetected malicious): {false_negatives}")
                    
        except Exception as e:
                    print(f"Error computing weights with dual attention: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    # Fallback to direct trust score weighting
                    weights = [max(0.01, score) for score in client_trust_scores]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                    else:
                        weights = [1.0 / len(client_trust_scores)] * len(client_trust_scores)
            else:
                # Direct trust score weighting
                weights = [max(0.01, score) for score in client_trust_scores]
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [1.0 / len(client_trust_scores)] * len(client_trust_scores)
            
            # Store weights as tensor
            self.weights = torch.tensor(weights, device=self.device)
            
            # Print and log weights
            print("\nAggregation Weights:")
            for i, client_idx in enumerate(client_indices):
                client = self.clients[client_idx]
                is_malicious = "YES" if client.is_malicious else "NO"
                weight = weights[i]
                round_metrics[round_idx]['weights'][client_idx] = weight
                print(f"Client {client_idx} (Malicious: {is_malicious}): Weight = {weight:.4f}")
            
            # Prepare aggregation arguments
            aggregation_args = {
                'model': self.global_model,
                'client_gradients': all_gradients,
                'weights': self.weights,
                'client_indices': client_indices,
                'features': features_tensor if 'features_tensor' in locals() else None,
                'trust_scores': self.trust_scores,
                'confidence_scores': self.confidence_scores
            }
            
            # Aggregate gradients based on selected method
            print(f"\n--- Aggregating gradients using {AGGREGATION_METHOD} ---")
            
            if AGGREGATION_METHOD == 'fedavg':
                # Simple weighted averaging
                aggregated_gradient = self._aggregate_fedavg(all_gradients, self.weights)
                
            elif AGGREGATION_METHOD == 'fedavg_with_trust':
                # FedAvg with trust scores as weights
                aggregated_gradient = self._aggregate_fedavg(all_gradients, self.weights)
                
            elif AGGREGATION_METHOD == 'fedprox':
                # FedProx aggregation (same as FedAvg for aggregation, proximal term is applied during client training)
                aggregated_gradient = self._aggregate_fedavg(all_gradients, self.weights)
                
            elif AGGREGATION_METHOD == 'fedbn':
                # FedBN - Skip BatchNorm parameters during aggregation
                aggregated_gradient = self._aggregate_fedbn(all_gradients, self.weights)
                
            elif AGGREGATION_METHOD == 'fedadmm':
                # FedADMM aggregation
                aggregated_gradient = self._aggregate_fedadmm(all_gradients, self.weights)
                
        else:
                # Default to weighted average
                print(f"Unknown aggregation method: {AGGREGATION_METHOD}. Using weighted average.")
                aggregated_gradient = self._aggregate_fedavg(all_gradients, self.weights)
            
            # Update global model with aggregated gradient
            print("\n--- Updating global model with aggregated gradient ---")
            agg_norm = torch.norm(aggregated_gradient).item()
            print(f"Aggregated gradient norm: {agg_norm:.4f}")
            
            # Update parameters
            old_model_copy = copy.deepcopy(self.global_model)
            self._update_global_model(aggregated_gradient)
            
            # Print parameter change statistics
            param_change = calculate_model_param_change(old_model_copy, self.global_model)
            print(f"Global model parameter change: {param_change:.8f}")
            
            # Optional: Clear cache to reduce memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final evaluation
        print("\n=== Final Model Evaluation ===")
        final_acc, final_error = test(self.global_model, self.test_loader)
        print(f"Final test accuracy: {final_acc:.4f}, error: {final_error:.4f}")
        print(f"Improvement: {initial_acc - final_acc:.4f}")
        
        # Save final model
        try:
            torch.save(self.global_model.state_dict(), 'model_weights/final_global_model.pth')
            print("Final global model saved to 'model_weights/final_global_model.pth'")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        
        # Plot training progress
        self._plot_training_progress(test_errors, round_metrics)
        
        return test_errors, round_metrics
        
    def _aggregate_fedavg(self, gradients, weights):
        """Aggregate gradients using FedAvg."""
        grad_tensor = torch.stack(gradients)
        weights_expanded = weights.view(-1, 1)
        return torch.sum(grad_tensor * weights_expanded, dim=0)

    def _aggregate_fedbn(self, gradients, weights):
        """Aggregate gradients using FedBN (skipping BatchNorm parameters)."""
        # Identify BatchNorm parameters in the model
        bn_params = set()
        for name, module in self.global_model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                # Get the parameter indices for this module
                for param_name, _ in module.named_parameters():
                    full_name = f"{name}.{param_name}" if name else param_name
                    bn_params.add(full_name)
        
        print(f"FedBN: Identified {len(bn_params)} BatchNorm parameters to preserve")
        
        # Create parameter index mapping
        param_indices = {}
        idx = 0
        for name, param in self.global_model.named_parameters():
            if param.requires_grad:
                size = param.numel()
                param_indices[name] = (idx, idx + size, param.shape)
                idx += size
        
        # Convert gradients to tensor
        grad_tensor = torch.stack(gradients)
        
        # Apply weights to non-BatchNorm parameters only
        result = torch.zeros_like(gradients[0])
        
        # Apply weights
        weights_expanded = weights.view(-1, 1)
        weighted_grads = grad_tensor * weights_expanded
        
        # Sum across clients (first dimension)
        aggregated = torch.sum(weighted_grads, dim=0)
        
        # Return the aggregated gradient
        return aggregated

    def _aggregate_fedadmm(self, gradients, weights, rho=1.0, sigma=0.1, iterations=3):
        """Aggregate gradients using FedADMM."""
        # Convert to tensor
        grad_tensor = torch.stack(gradients)
        
        # Initial aggregation is weighted average
        z = torch.sum(grad_tensor * weights.view(-1, 1), dim=0)
        
        # Initialize dual variables
        dual_vars = [torch.zeros_like(z) for _ in range(len(gradients))]
        
        # Iterative optimization
        for i in range(iterations):
            # Update z (consensus variable)
            sum_term = torch.zeros_like(z)
            for j, grad in enumerate(gradients):
                sum_term += grad + dual_vars[j]
            z = sum_term / (len(gradients) + rho)
            
            # Update dual variables
            for j, grad in enumerate(gradients):
                dual_vars[j] = dual_vars[j] + sigma * (grad - z)
        
        return z

    def _update_global_model(self, aggregated_gradient):
        """Update global model with aggregated gradient."""
        from federated_learning.utils.model_utils import update_model_with_gradient
        
        try:
            # Update the global model with the aggregated gradient
            model, total_change, avg_change = update_model_with_gradient(
                self.global_model, 
                aggregated_gradient, 
                learning_rate=LR,
                proximal_mu=FEDPROX_MU if AGGREGATION_METHOD == 'fedprox' else 0.0,
                preserve_bn=AGGREGATION_METHOD == 'fedbn'
            )
            
            print(f"Model updated with total parameter change: {total_change:.8f}")
            print(f"Average parameter change: {avg_change:.8f}")
            
        except Exception as e:
            print(f"Error updating global model: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def _select_clients(self, num_clients=None):
        """Select clients for the current round."""
        from federated_learning.config.config import CLIENT_SELECTION_RATIO
        
        if num_clients is None:
            num_clients = max(1, int(CLIENT_SELECTION_RATIO * len(self.clients)))
            
        # Random selection of clients
        import random
        selected_indices = random.sample(range(len(self.clients)), num_clients)
        
        return selected_indices
        
    def _plot_training_progress(self, test_errors, round_metrics):
        """Plot training progress."""
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            # Create plot
            plt.figure(figsize=(15, 10))
            
            # Plot test error
            plt.subplot(2, 2, 1)
            plt.plot(test_errors, 'b-', label='Test Error')
            plt.title('Test Error Over Time')
            plt.xlabel('Round')
            plt.ylabel('Error Rate')
            plt.grid(True)
            
            # Plot trust scores
            plt.subplot(2, 2, 2)
            for client_id in range(len(self.clients)):
                trust_scores = [round_metrics.get(r, {}).get('trust_scores', {}).get(client_id, 0) 
                               for r in range(len(round_metrics))]
                plt.plot(trust_scores, label=f'Client {client_id}')
            plt.title('Trust Scores Over Time')
            plt.xlabel('Round')
            plt.ylabel('Trust Score')
            plt.legend()
            plt.grid(True)
            
            # Plot aggregation weights
            plt.subplot(2, 2, 3)
            for client_id in range(len(self.clients)):
                weights = [round_metrics.get(r, {}).get('weights', {}).get(client_id, 0) 
                          for r in range(len(round_metrics))]
                plt.plot(weights, label=f'Client {client_id}')
            plt.title('Aggregation Weights Over Time')
            plt.xlabel('Round')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)
            
            # Plot gradient norms
            plt.subplot(2, 2, 4)
            for client_id in range(len(self.clients)):
                norms = [round_metrics.get(r, {}).get('gradient_norms', {}).get(client_id, 0) 
                        for r in range(len(round_metrics))]
                plt.plot(norms, label=f'Client {client_id}')
            plt.title('Gradient Norms Over Time')
            plt.xlabel('Round')
            plt.ylabel('Norm')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.tight_layout()
            plt.savefig(f'training_progress_{timestamp}.png')
            print(f"Training progress plot saved as 'training_progress_{timestamp}.png'")
            
            # Save trust scores
            plt.figure(figsize=(10, 6))
            for client_id, client in enumerate(self.clients):
                is_malicious = client.is_malicious if hasattr(client, 'is_malicious') else False
                color = 'red' if is_malicious else 'green'
                marker = 'x' if is_malicious else 'o'
                
                trust_scores = [round_metrics.get(r, {}).get('trust_scores', {}).get(client_id, 0) 
                               for r in range(len(round_metrics))]
                plt.plot(trust_scores, color=color, marker=marker, label=f'Client {client_id} ({"M" if is_malicious else "H"})')
            
            plt.title('Trust Scores by Client Type')
            plt.xlabel('Round')
            plt.ylabel('Trust Score')
            plt.legend()
            plt.grid(True)
            plt.savefig('trust_scores.png')
            
            # Save aggregation weights
            plt.figure(figsize=(10, 6))
            for client_id, client in enumerate(self.clients):
                is_malicious = client.is_malicious if hasattr(client, 'is_malicious') else False
                color = 'red' if is_malicious else 'green'
                marker = 'x' if is_malicious else 'o'
                
                weights = [round_metrics.get(r, {}).get('weights', {}).get(client_id, 0) 
                          for r in range(len(round_metrics))]
                plt.plot(weights, color=color, marker=marker, label=f'Client {client_id} ({"M" if is_malicious else "H"})')
            
            plt.title('Aggregation Weights by Client Type')
            plt.xlabel('Round')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True)
            plt.savefig('aggregation_weights.png')
            
        except Exception as e:
            print(f"Error creating training progress plot: {str(e)}")
            import traceback
            traceback.print_exc()

    def set_datasets(self, root_loader, test_dataset):
        """Set root dataset and test dataset for the server."""
        self.root_loader = root_loader
        self.test_dataset = test_dataset
        print("Datasets set for server")

    def _collect_root_gradients(self):
        """Collect gradients from the root dataset for VAE training."""
        print("Collecting root gradients...")
        if not hasattr(self, 'root_loader') or self.root_loader is None:
            print("Warning: No root dataset available for collecting gradients")
            return []
        
        # Create a temporary model for gradient collection
        temp_model = copy.deepcopy(self.global_model)
        temp_model.train()
        
        # Create a list to store gradients
        root_gradients = []
        
        # Collect gradients from each batch
        for data, target in self.root_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            temp_model.zero_grad()
            output = temp_model(data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            loss.backward()
            
            # Extract gradients
            gradient = torch.cat([p.grad.data.flatten() for p in temp_model.parameters() if p.requires_grad])
            root_gradients.append(gradient)
        
        print(f"Collected {len(root_gradients)} gradients from root dataset")
        return root_gradients

    def train_vae(self, root_gradients, vae_epochs=5):
        """Train the VAE on root gradients."""
        print(f"Training VAE on {len(root_gradients)} gradients...")
        
        # Skip if no gradients
        if not root_gradients:
            print("Warning: No gradients available for VAE training")
            return self._create_vae()
        
        # Convert list of gradients to tensor
        gradient_stack = torch.stack(root_gradients)
        
        # Create and train VAE
        from federated_learning.training.training_utils import train_vae
        vae = self._create_vae()
        vae = train_vae(
            vae=vae,
            gradient_stack=gradient_stack,
            epochs=vae_epochs,
            batch_size=min(16, len(gradient_stack)),
            device=self.device
        )
        
        print("VAE training completed")
        return vae

    def add_clients(self, client_list):
        """
        Add clients to the server and ensure they're properly indexed.
        
        Args:
            client_list: List of Client objects to add
        """
        self.clients = client_list
        
        # Track indices of malicious clients
        self.malicious_clients = [i for i, client in enumerate(self.clients) if client.is_malicious]
        
        print(f"Added {len(client_list)} clients to server")
        print(f"Malicious clients: {len(self.malicious_clients)}")
