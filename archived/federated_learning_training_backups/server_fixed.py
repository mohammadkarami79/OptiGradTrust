import torch
import torch.optim as optim
import numpy as np
import random
import copy
from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.models.resnet import ResNet50Alzheimer, ResNet18Alzheimer
from federated_learning.models.vae import GradientVAE
from federated_learning.models.attention import DualAttention
from federated_learning.models.dimension_reducer import GradientDimensionReducer
from federated_learning.training.training_utils import train_vae, train_dual_attention, test
from federated_learning.training.client import Client
from federated_learning.data.dataset import (
    LabelFlippingDataset, BackdoorDataset, AdaptiveAttackDataset,
    MinMaxAttackDataset, MinSumAttackDataset, AlternatingAttackDataset, 
    TargetedAttackDataset, GradientInversionAttackDataset
)

class Server:
    def __init__(self, root_dataset, client_datasets, test_dataset, num_classes=None):
        print("\n=== Initializing Server ===")
        
        # Determine number of classes
        if num_classes is None:
            if DATASET == 'MNIST':
                self.num_classes = 10
            elif DATASET == 'ALZHEIMER':
                self.num_classes = ALZHEIMER_CLASSES
            else:
                raise ValueError(f"Unknown dataset: {DATASET}")
        else:
            self.num_classes = num_classes
            
        print("Creating global model...")
        # Initialize the appropriate model based on config
        if MODEL == 'CNN':
            self.global_model = CNNMnist().to(device)
        elif MODEL == 'RESNET50':
            self.global_model = ResNet50Alzheimer(
                num_classes=self.num_classes,
                unfreeze_layers=RESNET50_UNFREEZE_LAYERS,
                pretrained=RESNET_PRETRAINED
            ).to(device)
        elif MODEL == 'RESNET18':
            self.global_model = ResNet18Alzheimer(
                num_classes=self.num_classes,
                unfreeze_layers=RESNET18_UNFREEZE_LAYERS,
                pretrained=RESNET_PRETRAINED
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {MODEL}")
            
        self.test_dataset = test_dataset
        self.root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize gradient dimension reducer if enabled
        self.dimension_reducer = None
        if ENABLE_DIMENSION_REDUCTION and (MODEL == 'RESNET50' or MODEL == 'RESNET18'):
            print(f"\n=== Initializing Gradient Dimension Reducer ===")
            print(f"Reduction ratio: {DIMENSION_REDUCTION_RATIO}")
            self.dimension_reducer = GradientDimensionReducer(reduction_ratio=DIMENSION_REDUCTION_RATIO)
            print("Gradient dimension reducer initialized. This will reduce memory usage for large models.")
        
        print("\nInitializing clients...")
        self.clients = []
        malicious_clients = random.sample(range(NUM_CLIENTS), NUM_MALICIOUS)
        print("\n=== Malicious Clients ===")
        print(f"Malicious clients: {malicious_clients}")
        print("=== Client Information ===")
        
        # Initialize clients with appropriate dataset wrappers for malicious clients
        for i in range(NUM_CLIENTS):
            is_malicious = i in malicious_clients
            
            # If client is malicious, wrap its dataset with appropriate attack dataset
            if is_malicious:
                if ATTACK_TYPE == 'label_flipping':
                    wrapped_dataset = LabelFlippingDataset(client_datasets[i], self.num_classes)
                elif ATTACK_TYPE == 'backdoor_attack':
                    wrapped_dataset = BackdoorDataset(client_datasets[i], self.num_classes)
                elif ATTACK_TYPE == 'adaptive_attack':
                    wrapped_dataset = AdaptiveAttackDataset(client_datasets[i])
                elif ATTACK_TYPE == 'min_max_attack':
                    wrapped_dataset = MinMaxAttackDataset(client_datasets[i], self.num_classes)
                elif ATTACK_TYPE == 'min_sum_attack':
                    wrapped_dataset = MinSumAttackDataset(client_datasets[i], self.num_classes)
                elif ATTACK_TYPE == 'alternating_attack':
                    wrapped_dataset = AlternatingAttackDataset(client_datasets[i], self.num_classes)
                elif ATTACK_TYPE == 'targeted_attack':
                    wrapped_dataset = TargetedAttackDataset(client_datasets[i], self.num_classes)
                elif ATTACK_TYPE == 'gradient_inversion_attack':
                    wrapped_dataset = GradientInversionAttackDataset(client_datasets[i], self.num_classes)
                else:
                    # For other attacks (scaling_attack, partial_scaling_attack), use original dataset
                    wrapped_dataset = client_datasets[i]
                
                self.clients.append(Client(i, wrapped_dataset, is_malicious, self.num_classes))
            else:
                self.clients.append(Client(i, client_datasets[i], is_malicious, self.num_classes))
                
            print(f"Client {i}: {'MALICIOUS' if is_malicious else 'HONEST'}, Dataset size: {len(client_datasets[i])}")
        
        print("\nInitializing VAE and Dual Attention...")
        # Use VAE_DEVICE configuration instead of hardcoding to CPU
        if VAE_DEVICE == 'gpu' and torch.cuda.is_available():
            vae_device = torch.device('cuda')
            print(f"VAE will run on GPU (CUDA) as specified in configuration")
        elif VAE_DEVICE == 'auto':
            # Auto mode: Use GPU if available and main model is on CPU, otherwise use CPU
            if device.type == 'cpu' and torch.cuda.is_available():
                vae_device = torch.device('cuda')
                print(f"VAE will run on GPU (auto mode - main model on CPU, GPU available)")
            else:
                vae_device = torch.device('cpu')
                print(f"VAE will run on CPU (auto mode - saving GPU memory for main model)")
        else:
            vae_device = torch.device('cpu')
            print(f"VAE will run on CPU (explicitly configured or GPU not available)")
        
        # Initialize VAE with new parameter name (input_dim instead of input_size)
        param_count = sum(p.numel() for p in self.global_model.parameters())
        print(f"Model has {param_count} parameters")
        
        # Use projection for memory efficiency - automatically enabled for large models
        # For very large models, use a smaller projection dimension (1024 instead of 2048)
        projection_dim = 1024 if param_count > 10_000_000 else 2048
        
        self.vae = GradientVAE(
            input_dim=param_count, 
            hidden_dim=512,  # Reduce hidden dimensions for memory efficiency
            latent_dim=128,  # Reduce latent dimensions for memory efficiency
            dropout_rate=0.2,
            projection_dim=projection_dim
        ).to(vae_device)
        
        self.dual_attention = DualAttention(feature_size=4).to(device)
        
        print("\nCollecting root gradients...")
        self.root_gradients = self._collect_root_gradients()
        print(f"Collected {len(self.root_gradients)} root gradients")
        
        print("\nTraining VAE and Dual Attention...")
        self._train_models()

    def _collect_root_gradients(self):
        print("\n=== Collecting Root Gradients ===")
        root_gradients = []
        global_model_copy = copy.deepcopy(self.global_model)
        criterion = torch.nn.NLLLoss()
        
        # Configure dataloader with optimal parameters for GPU
        self.root_loader = torch.utils.data.DataLoader(
            self.root_loader.dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=NUM_WORKERS if torch.cuda.is_available() else 0,
            pin_memory=PIN_MEMORY if torch.cuda.is_available() else False
        )
        
        # Display root dataset information
        if hasattr(self.root_loader.dataset, 'indices'):
            root_dataset_size = len(self.root_loader.dataset.indices)
        else:
            root_dataset_size = len(self.root_loader.dataset)
        print(f"Root dataset size: {root_dataset_size} samples")
        print(f"Batch size: {self.root_loader.batch_size}")
        print(f"Number of batches per epoch: {len(self.root_loader)}")
        print(f"Expected gradients to collect: {LOCAL_EPOCHS_ROOT}")
        print(f"Gradient chunking: {GRADIENT_CHUNK_SIZE} epochs per chunk using '{GRADIENT_AGGREGATION_METHOD}' aggregation")
        print(f"Total chunks to collect: {LOCAL_EPOCHS_ROOT // GRADIENT_CHUNK_SIZE + (1 if LOCAL_EPOCHS_ROOT % GRADIENT_CHUNK_SIZE > 0 else 0)}")
        
        # GPU memory info before training
        if torch.cuda.is_available():
            print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
        
        # Display dimension reduction info if enabled
        if self.dimension_reducer is not None:
            print(f"\n=== Dimension Reduction Enabled ===")
            print(f"Using PCA to reduce gradient dimensions")
            print(f"Reduction ratio: {DIMENSION_REDUCTION_RATIO}")
            
        try:
            # Variables needed for chunking
            current_chunk_gradients = []
            current_chunk_size = 0
            total_chunks_collected = 0
        
            for epoch in range(LOCAL_EPOCHS_ROOT):
                print(f"\nProcessing epoch {epoch+1}/{LOCAL_EPOCHS_ROOT}...")
                
                # Create a new copy of the original model for each epoch
                root_model = copy.deepcopy(self.global_model)
                root_optimizer = optim.SGD(
                    root_model.parameters(), 
                    lr=LR,
                    momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY
                )
                
                epoch_loss = 0
                batches = 0
                
                # Train the model on all batches in this epoch
                print(f"Training on {len(self.root_loader)} batches...")
                
                for batch_idx, (data, target) in enumerate(self.root_loader):
                    if batch_idx % 20 == 0:
                        print(f"  Processing batch {batch_idx+1}/{len(self.root_loader)}...")
                    
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    root_optimizer.zero_grad(set_to_none=True)  # Memory optimization
                    
                    try:
                        output = root_model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        root_optimizer.step()
                        
                        epoch_loss += loss.item()
                        batches += 1
                    except Exception as e:
                        print(f"Error in batch {batch_idx+1}: {str(e)}")
                        print(f"Data shape: {data.shape}, Target shape: {target.shape}")
                        continue
                    
                    # Clean up memory
                    del data, target, output, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if batches == 0:
                    print("Warning: No batches were processed in this epoch!")
                    continue
                    
                avg_loss = epoch_loss / batches
                print(f"Epoch completed. Average loss: {avg_loss:.4f}")
                
                # After complete training in this epoch, compute gradient (difference between trained model and original model)
                print("Computing gradient (difference between trained model and original model)...")
                grad_list = []
                
                for (name, pg), (_, plg) in zip(global_model_copy.named_parameters(), root_model.named_parameters()):
                    diff = (plg.data - pg.data).view(-1)
                    grad_list.append(diff)
                    
                    # Display statistical information for each layer
                    if epoch == 0 and len(root_gradients) == 0:
                        print(f"  Layer: {name}, Shape: {pg.shape}, Diff Norm: {torch.norm(diff):.6f}")
                
                # Create gradient vector
                grad_vector = torch.cat(grad_list).detach()
                
                # Normalize gradient
                norm_val = torch.norm(grad_vector) + 1e-8
                normalized_grad = grad_vector / norm_val
                
                # Check gradient validity
                if torch.isnan(normalized_grad).any():
                    print("Warning: NaN values detected in gradient!")
                if torch.isinf(normalized_grad).any():
                    print("Warning: Inf values detected in gradient!")
                
                # Reduce gradient dimensions using dimension reducer if enabled
                if self.dimension_reducer is not None:
                    # If this is the first gradient, prepare the dimension reducer
                    if epoch == 0 and not hasattr(self.dimension_reducer, 'is_fitted'):
                        print("Fitting dimension reducer to first gradient...")
                        self.dimension_reducer.fit([normalized_grad])
                    
                    # Reduce gradient dimensions
                    print("Applying dimension reduction...")
                    reduced_grad = self.dimension_reducer.transform([normalized_grad])[0]
                    print(f"Reduced gradient dimensions from {normalized_grad.shape} to {reduced_grad.shape}")
                    
                    # Replace original gradient with reduced gradient
                    normalized_grad = reduced_grad
                
                # Add gradient to current chunk
                current_chunk_gradients.append(normalized_grad)
                current_chunk_size += 1
                
                # Check if chunk is complete
                if current_chunk_size >= GRADIENT_CHUNK_SIZE or epoch == LOCAL_EPOCHS_ROOT - 1:
                    # Aggregate chunk gradients using specified method
                    if GRADIENT_AGGREGATION_METHOD == 'mean':
                        # Average gradients
                        chunk_gradient = torch.stack(current_chunk_gradients).mean(dim=0)
                    elif GRADIENT_AGGREGATION_METHOD == 'sum':
                        # Sum gradients
                        chunk_gradient = torch.stack(current_chunk_gradients).sum(dim=0)
                        # Re-normalize
                        chunk_gradient = chunk_gradient / (torch.norm(chunk_gradient) + 1e-8)
                    elif GRADIENT_AGGREGATION_METHOD == 'last':
                        # Just the last gradient
                        chunk_gradient = current_chunk_gradients[-1]
                    else:
                        # Default: average
                        chunk_gradient = torch.stack(current_chunk_gradients).mean(dim=0)
                    
                    # Add to the list of root gradients
                    root_gradients.append(chunk_gradient)
                    total_chunks_collected += 1
                    
                    # Display chunk information
                    print(f"Chunk {total_chunks_collected} collected with {current_chunk_size} gradients using '{GRADIENT_AGGREGATION_METHOD}' method.")
                    
                    # Reset chunk variables
                    current_chunk_gradients = []
                    current_chunk_size = 0
                
                if (epoch + 1) % 10 == 0:
                    print(f"Root Training Epoch {epoch + 1}/{LOCAL_EPOCHS_ROOT}, Loss: {avg_loss:.4f}")
                    if torch.cuda.is_available():
                        print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
                
                # Clean up memory
                del root_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"\nCollection completed. Collected {len(root_gradients)} gradient chunks from {LOCAL_EPOCHS_ROOT} epochs")
            return root_gradients
        except Exception as e:
            print(f"Error in _collect_root_gradients: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return at least the gradients collected so far in case of error
            print(f"Returning {len(root_gradients)} gradients collected so far")
            return root_gradients

    def _train_models(self):
        """Train the VAE and Dual Attention models with collected root gradients"""
        print("\n=== Training VAE Model ===")
        try:
            # Prepare gradients - ensure they are normalized
            normalized_gradients = []
            for grad in self.root_gradients:
                if isinstance(grad, torch.Tensor):
                    norm = torch.norm(grad) + 1e-8
                    normalized_gradients.append(grad / norm)
                else:
                    print("Warning: Non-tensor gradient found. Skipping.")
            
            if len(normalized_gradients) == 0:
                print("Error: No valid gradients for training. Check gradient collection.")
                return
                
            print(f"Training VAE on {len(normalized_gradients)} normalized gradients")
            
            # Train the VAE model
            train_vae(self.vae, normalized_gradients)
            
            print("VAE training completed.")
            
            # For dual attention training, we need labeled data which we'll get during FL rounds
            print("\n=== Dual Attention Training Prepared ===")
            print("Dual Attention will be trained during federated learning rounds\n")
            
        except Exception as e:
            print(f"Error in _train_models: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def _compute_gradient_features(self, grad, raw_grad):
        """
        Compute feature vector for a gradient.
        
        Args:
            grad: Normalized gradient
            raw_grad: Raw gradient before normalization
            
        Returns:
            List of features [RE_val, mean_cosine_root, mean_neighbor_sim, grad_norm]
        """
        # Determine the device used by the VAE
        vae_device = next(self.vae.parameters()).device
        
        # Transfer gradient to the VAE's device for computation
        vae_grad = grad.to(vae_device)
        
        # Compute reconstruction error using VAE
        grad_input = vae_grad.unsqueeze(0)
        
        # Handle different return value formats
        outputs = self.vae(grad_input)
        if len(outputs) == 4:
            recon_batch, mu, logvar, _ = outputs
        else:
            recon_batch, mu, logvar = outputs
            
        RE_val = torch.nn.functional.mse_loss(recon_batch, grad_input, reduction='sum').item()
        
        # Compute mean cosine similarity with root gradients
        cos_root_vals = [torch.nn.functional.cosine_similarity(grad, r, dim=0).item() 
                        for r in self.root_gradients]
        mean_cosine_root = np.mean(cos_root_vals)
        
        # Compute norm of raw gradient
        grad_norm = torch.norm(raw_grad).item()
        
        # For mean_neighbor_sim, we can't compute it properly here
        # since we don't have gradients from other clients yet
        # Use a placeholder value that will be updated during the global training
        mean_neighbor_sim = 0.0
        
        # Free temporary memory
        del vae_grad, grad_input, recon_batch, mu, logvar
        
        return [RE_val, mean_cosine_root, mean_neighbor_sim, grad_norm] 