import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import random
from federated_learning.config.config import *
from federated_learning.config import config  # Import as module for runtime access

# Import OASIS classes constant
try:
    OASIS_CLASSES = 4  # OASIS-1 has 4 dementia classes (CDR 0, 0.5, 1, 2)
except:
    OASIS_CLASSES = 4
from federated_learning.training.training_utils import client_update
from federated_learning.models.cnn import CNNMnist
from federated_learning.models.resnet import ResNet50Alzheimer, ResNet18Alzheimer
from federated_learning.models.dimension_reducer import GradientDimensionReducer
from federated_learning.attacks.attack_utils import simulate_attack
from federated_learning.privacy.privacy_utils import apply_privacy_mechanism
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate, Dataset
import math

class SubsetDataset(Dataset):
    """Custom dataset wrapper for Subset objects to avoid pickling issues."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def subset_collate_fn(batch, dataset):
    """Collate function for handling Subset objects in DataLoader."""
    indices = [b for b in range(len(batch))]
    return default_collate([dataset.dataset[dataset.indices[idx]] for idx in indices])

class Attack:
    """Class for implementing various attacks in federated learning."""
    
    def __init__(self, attack_type):
        """Initialize attack with the specified type."""
        self.attack_type = attack_type
    
    def apply_attack(self, data, target):
        """Apply attack to input data and target."""
        # For most attack types, we only attack the gradient, not the data
        # But some attacks like label flipping work on the data/labels
        if self.attack_type == 'label_flipping':
            # Flip labels for classification tasks
            num_classes = len(torch.unique(target))
            # Handle edge case: if only 1 class in batch, we need at least 2 for flipping
            if num_classes < 2:
                # Assume 4 classes for medical datasets (typical), otherwise use 10
                num_classes = 4 if hasattr(self, 'num_classes') else 4
                print(f"Label flipping: Only 1 class in batch, using {num_classes} classes for flipping")
            new_target = (target + torch.randint(1, num_classes, target.shape, device=target.device)) % num_classes
            print(f"Label flipping attack: Flipped {(new_target != target).sum().item()}/{len(target)} labels")
            return data, new_target
        else:
            # No data attack for other attack types
            return data, target
    
    def apply_gradient_attack(self, gradient):
        """Apply attack to gradient based on attack type."""
        device = gradient.device
        
        # Store original values for reporting
        original_gradient = gradient.clone()
        original_norm = torch.norm(gradient).item()
        
        # Get stored attack parameters or use defaults
        scaling_factor = getattr(self, 'scaling_factor', SCALING_FACTOR if 'SCALING_FACTOR' in globals() else 15.0)
        partial_percent = getattr(self, 'partial_percent', PARTIAL_SCALING_PERCENT if 'PARTIAL_SCALING_PERCENT' in globals() else 0.3)
        
        # Use parameters from attack_params if available
        if hasattr(self, 'attack_params'):
            scaling_factor = self.attack_params.get('scaling_factor', scaling_factor)
            partial_percent = self.attack_params.get('partial_percent', partial_percent)
        
        # Apply attack based on type
        if self.attack_type == 'scaling_attack':
            # Scale the entire gradient by a large factor
            modified_grad = gradient * scaling_factor
            attack_name = f"Scaling attack (factor: {scaling_factor:.1f})"
            
        elif self.attack_type == 'sign_flipping_attack':
            # Flip the signs of the gradient
            modified_grad = -gradient
            attack_name = "Sign flipping attack"
            
        elif self.attack_type == 'partial_scaling_attack':
            # Scale a portion of the gradient elements
            # Select portion of the gradient elements to scale based on config
            mask = torch.zeros_like(gradient, device=device)
            indices = torch.randperm(gradient.numel(), device=device)[:int(partial_percent * gradient.numel())]
            mask.view(-1)[indices] = 1.0
            
            # Apply scaling only to selected elements
            modified_grad = gradient.clone()
            modified_grad = modified_grad * (1 - mask) + modified_grad * mask * scaling_factor
            attack_name = f"Partial scaling attack (factor: {scaling_factor:.1f}, {partial_percent*100:.1f}% of elements)"
            
        elif self.attack_type == 'noise_attack':
            # Add Gaussian noise; use config/revision severity (noise_factor) when set
            noise_factor = getattr(self, 'noise_factor', None)
            if noise_factor is None and hasattr(self, 'attack_params'):
                noise_factor = self.attack_params.get('noise_factor')
            if noise_factor is None:
                noise_factor = 0.5
            grad_std = torch.std(gradient).item() or 1e-6
            noise = torch.randn_like(gradient, device=device) * grad_std * noise_factor
            modified_grad = gradient + noise
            attack_name = f"Noise attack (factor: {noise_factor:.1f})"

        elif self.attack_type == 'gaussian_noise_injection':
            # Gaussian noise injection with absolute sigma (not relative to gradient std)
            # Used in R2 ablation experiments (sigma=15.0 by default)
            sigma = getattr(self, 'sigma', 15.0)
            if hasattr(self, 'attack_params') and self.attack_params:
                sigma = self.attack_params.get('sigma', sigma)
            noise = torch.randn_like(gradient, device=device) * sigma
            modified_grad = gradient + noise
            attack_name = f"Gaussian noise injection (sigma={sigma:.1f})"

        elif self.attack_type == 'adaptive_vae_attack':
            # R3-D: VAE-aware adaptive attack.
            # PGD inner loop in gradient space that minimises
            #     loss = alpha * ||VAE(g') - g'||^2 + beta * (1 - cos(g', target))
            # where target = -gradient * scaling_factor (the malicious direction).
            # Runs only if the server has injected a VAE snapshot into
            # self.vae_snapshot; otherwise falls back to stealth-scaling.
            factor = getattr(self, 'scaling_factor', scaling_factor)
            if hasattr(self, 'attack_params') and self.attack_params:
                factor = self.attack_params.get('scaling_factor', factor)
            stealth_ratio = (self.attack_params.get('stealth_ratio', 0.5)
                             if hasattr(self, 'attack_params') and self.attack_params else 0.5)
            n_steps = (self.attack_params.get('adaptive_steps', 20)
                       if hasattr(self, 'attack_params') and self.attack_params else 20)
            step_size = (self.attack_params.get('adaptive_step_size', 0.05)
                         if hasattr(self, 'attack_params') and self.attack_params else 0.05)
            alpha_vae = (self.attack_params.get('alpha_vae', 1.0)
                         if hasattr(self, 'attack_params') and self.attack_params else 1.0)
            beta_dir = (self.attack_params.get('beta_dir', 0.5)
                        if hasattr(self, 'attack_params') and self.attack_params else 0.5)

            vae_snap = getattr(self, 'vae_snapshot', None)
            target = -gradient * factor
            eps_radius = 5.0 * original_norm

            if vae_snap is not None:
                # PGD in gradient space against the VAE
                try:
                    vae_snap.eval()
                    for p in vae_snap.parameters():
                        p.requires_grad_(False)

                    g_adv = gradient.clone().detach()
                    for _ in range(n_steps):
                        g_adv = g_adv.detach().requires_grad_(True)
                        recon, _, _ = vae_snap(g_adv.unsqueeze(0))
                        recon = recon.squeeze(0)
                        recon_loss = torch.mean((recon - g_adv) ** 2)
                        cos = F.cosine_similarity(g_adv.unsqueeze(0),
                                                  target.unsqueeze(0)).squeeze()
                        align_loss = 1.0 - cos
                        loss = alpha_vae * recon_loss + beta_dir * align_loss
                        g_grad = torch.autograd.grad(loss, g_adv,
                                                    retain_graph=False,
                                                    create_graph=False)[0]
                        g_adv = g_adv.detach() - step_size * g_grad
                        # project into an L2 ball of radius eps_radius around the benign gradient
                        delta = g_adv - gradient
                        dn = torch.norm(delta).item() + 1e-12
                        if dn > eps_radius:
                            delta = delta * (eps_radius / dn)
                            g_adv = gradient + delta

                    modified_grad = g_adv.detach()
                    attack_name = (f"Adaptive VAE attack "
                                   f"(PGD, steps={n_steps}, factor={factor:.1f})")
                except Exception as _exc:
                    # Graceful fallback if the VAE forward fails mid-attack
                    print(f"[adaptive_vae_attack] VAE-PGD failed ({_exc}); "
                          "falling back to stealth-scaling")
                    blended = (1.0 - stealth_ratio) * gradient + stealth_ratio * target
                    bn = torch.norm(blended) + 1e-12
                    on = torch.norm(gradient) + 1e-12
                    if bn > 2.0 * on:
                        blended = blended * (2.0 * on / bn)
                    modified_grad = blended
                    attack_name = (f"Adaptive VAE attack (fallback stealth, "
                                   f"ratio={stealth_ratio:.2f})")
            else:
                # No VAE available → stealth-scaling variant (still adaptive vs. norm-based detection)
                blended = (1.0 - stealth_ratio) * gradient + stealth_ratio * target
                bn = torch.norm(blended) + 1e-12
                on = torch.norm(gradient) + 1e-12
                if bn > 2.0 * on:
                    blended = blended * (2.0 * on / bn)
                modified_grad = blended
                attack_name = (f"Adaptive VAE attack (no VAE; stealth-scaling, "
                               f"ratio={stealth_ratio:.2f})")
            
        elif self.attack_type == 'min_max_attack':
            # Amplify largest elements and reduce smallest ones
            modified_grad = gradient.clone()
            
            # Find the top 10% largest (by magnitude) gradient elements
            num_elements = gradient.numel()
            values, indices = torch.topk(torch.abs(gradient.view(-1)), k=int(0.1 * num_elements))
            
            # Amplify by 3x
            modified_grad.view(-1)[indices] *= 3.0
            
            # Find bottom 50% elements
            values, indices = torch.topk(torch.abs(gradient.view(-1)), k=int(0.5 * num_elements), largest=False)
            
            # Reduce to 10% of original
            modified_grad.view(-1)[indices] *= 0.1
            attack_name = "Min-Max attack (amplify 10% largest, reduce 50% smallest)"
            
        elif self.attack_type == 'targeted_attack':
            # Targeted poisoning focusing on specific components
            modified_grad = gradient.clone()
            
            # Add noise to disguise the attack
            random_noise = torch.randn_like(modified_grad) * 0.2 * gradient.norm()
            modified_grad += random_noise
            
            # Target specific components by inverting and amplifying
            indices = torch.randperm(gradient.numel(), device=device)[:int(0.1 * gradient.numel())]
            modified_grad.view(-1)[indices] *= -3.0
            attack_name = "Targeted attack (selective inversion)"
            
        elif self.attack_type == 'backdoor_attack':
            # Subtle backdoor attack that's hard to detect
            modified_grad = gradient.clone()
            
            # Select a small subset (5%) of gradient components to modify
            mask = torch.zeros_like(gradient, device=device)
            indices = torch.randperm(gradient.numel(), device=device)[:int(0.05 * gradient.numel())]
            mask.view(-1)[indices] = 1.0
            
            # Create consistent pattern
            backdoor_pattern = torch.ones_like(gradient, device=device) * 0.1 * gradient.norm().item()
            modified_grad = modified_grad * (1 - mask) + backdoor_pattern * mask
            attack_name = "Backdoor attack (subtle pattern injection)"
            
        elif self.attack_type == 'label_flipping':
            # For label flipping, gradient is already poisoned during data/label modification
            # Return original gradient since the poisoning happens at data level
            modified_grad = gradient.clone()
            attack_name = "Label flipping attack (applied during training)"
            
        elif self.attack_type == 'min_sum_attack':
            # Minimize sum of targeted gradient components
            modified_grad = gradient.clone()
            
            # Target 20% of largest gradient components and invert them
            num_elements = gradient.numel()
            values, indices = torch.topk(torch.abs(gradient.view(-1)), k=int(0.2 * num_elements))
            
            # Invert and amplify
            modified_grad.view(-1)[indices] *= -2.0
            attack_name = "Min-Sum attack (invert 20% largest components)"
            
        else:
            # Default: no attack
            print(f"Unknown attack type: {self.attack_type}. Using original gradient.")
            return gradient
        
        # Calculate and print attack metrics
        modified_norm = torch.norm(modified_grad).item()
        norm_change_abs = modified_norm - original_norm
        # Avoid division by zero when original_norm is 0
        if original_norm > 1e-10:
            norm_change_pct = (modified_norm / original_norm - 1) * 100
        else:
            norm_change_pct = 0.0 if modified_norm < 1e-10 else float('inf')
        
        # Robust cosine similarity calculation - handle zero-norm vectors
        if original_norm > 1e-10 and modified_norm > 1e-10:
            cosine_sim = torch.nn.functional.cosine_similarity(
                modified_grad.view(1, -1), 
                original_gradient.view(1, -1)
            ).item()
        elif original_norm < 1e-10 and modified_norm < 1e-10:
            # Both zero vectors - consider them perfectly aligned
            cosine_sim = 1.0
        else:
            # One is zero, the other is not - consider them orthogonal
            cosine_sim = 0.0
        
        print(f"{attack_name} applied:")
        print(f"  Original norm: {original_norm:.4f}")
        print(f"  Modified norm: {modified_norm:.4f}")
        print(f"  Absolute change: {norm_change_abs:.4f}")
        print(f"  Percentage change: {norm_change_pct:.2f}%")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        
        return modified_grad

class Client:
    """
    Client class that models a client in federated learning.
    Each client has its own local data and can train a model.
    """
    
    def __init__(self, client_id, dataset, is_malicious=False, num_classes=None, local_epochs=None):
        """
        Initialize a client.
        
        Args:
            client_id: Unique client identifier
            dataset: Client's local dataset
            is_malicious: Whether this client is malicious
            num_classes: Number of model classes
            local_epochs: Number of local training epochs (uses default if None)
        """
        self.client_id = client_id
        self.dataset = dataset
        self.is_malicious = is_malicious
        self.num_classes = num_classes
        # CRITICAL: Read LOCAL_EPOCHS_CLIENT from config at runtime
        default_local_epochs = getattr(config, 'LOCAL_EPOCHS_CLIENT', LOCAL_EPOCHS_CLIENT)
        self.local_epochs = local_epochs if local_epochs is not None else default_local_epochs
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize metrics tracking with enhanced metrics
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'gradient_norms': [],
            'parameter_changes': [],
            'trust_scores': [],
            'confidence_scores': [],
            'feature_stats': {
                'mean': [],
                'std': [],
                'skewness': [],
                'kurtosis': []
            }
        }
        
        # Initialize model based on config
        # CRITICAL: Read from config module at runtime for dynamic dataset/model support
        current_model = getattr(config, 'MODEL', MODEL)
        current_dataset = getattr(config, 'DATASET', DATASET)
        alzheimer_classes = getattr(config, 'ALZHEIMER_CLASSES', ALZHEIMER_CLASSES)
        
        if current_model == 'CNN':
            # Determine input channels and number of classes based on dataset
            if current_dataset == 'MNIST':
                in_channels = 1
                num_classes = 10
            elif current_dataset == 'CIFAR10':
                in_channels = 3
                num_classes = 10
            elif current_dataset == 'ALZHEIMER':
                in_channels = 3
                num_classes = alzheimer_classes
            elif current_dataset == 'OASIS':
                in_channels = 3
                num_classes = 4  # OASIS CDR classes
            else:
                # Default fallback
                in_channels = 1
                num_classes = 10
                
            self.model = CNNMnist(in_channels=in_channels, num_classes=num_classes).to(self.device)
            
        elif current_model == 'ResNet50':
            if num_classes is None:
                if current_dataset == 'MNIST':
                    num_classes = 10
                elif current_dataset == 'CIFAR10':
                    num_classes = 10
                elif current_dataset == 'ALZHEIMER':
                    num_classes = alzheimer_classes
                elif current_dataset == 'OASIS':
                    num_classes = 4  # OASIS CDR classes
                else:
                    # Default to 4 classes for unknown medical datasets
                    num_classes = 4
                    print(f"Warning: Unknown dataset {current_dataset}, using {num_classes} classes")
            
            self.model = ResNet50Alzheimer(
                num_classes=num_classes,
                unfreeze_layers=RESNET50_UNFREEZE_LAYERS,
                pretrained=RESNET_PRETRAINED
            ).to(self.device)
            
        elif current_model == 'ResNet18':
            if num_classes is None:
                if current_dataset == 'MNIST':
                    num_classes = 10
                elif current_dataset == 'CIFAR10':
                    num_classes = 10
                elif current_dataset == 'ALZHEIMER':
                    num_classes = alzheimer_classes
                elif current_dataset == 'OASIS':
                    num_classes = 4  # OASIS CDR classes
                else:
                    # Default to 4 classes for unknown medical datasets
                    num_classes = 4
                    print(f"Warning: Unknown dataset {current_dataset}, using {num_classes} classes")
            
            self.model = ResNet18Alzheimer(
                num_classes=num_classes,
                unfreeze_layers=RESNET18_UNFREEZE_LAYERS,
                pretrained=RESNET_PRETRAINED
            ).to(self.device)
            
        else:
            raise ValueError(f"Unknown model type: {current_model}")
        
        # CRITICAL: Read LR and BATCH_SIZE from config at runtime
        current_lr = getattr(config, 'LR', LR)
        current_batch_size = getattr(config, 'BATCH_SIZE', BATCH_SIZE)
            
        self.optimizer = optim.SGD(self.model.parameters(), lr=current_lr)
        self.criterion = nn.NLLLoss()
        
        # Create proper dataloader for the dataset
        # If the dataset is a Subset, we need to wrap it in our custom SubsetDataset
        if isinstance(dataset, torch.utils.data.Subset):
            # Create a custom dataset wrapper
            custom_dataset = SubsetDataset(dataset.dataset, dataset.indices)
            self.train_loader = torch.utils.data.DataLoader(
                custom_dataset, 
                batch_size=current_batch_size, 
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False
            )
        else:
            # Standard dataloader for regular datasets
            self.train_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=current_batch_size, 
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False
            )
        
        # Initialize gradient dimension reducer if enabled
        self.dimension_reducer = None
        enable_dim_reduction = getattr(config, 'ENABLE_DIMENSION_REDUCTION', ENABLE_DIMENSION_REDUCTION)
        dim_reduction_ratio = getattr(config, 'DIMENSION_REDUCTION_RATIO', DIMENSION_REDUCTION_RATIO)
        if enable_dim_reduction and (current_model == 'ResNet50' or current_model == 'ResNet18'):
            self.dimension_reducer = GradientDimensionReducer(reduction_ratio=dim_reduction_ratio)
            print(f"Client {client_id}: Dimension reducer initialized with ratio {dim_reduction_ratio}")

        # Initialize attack if client is malicious
        if self.is_malicious:
            # Try to use ATTACK_TYPE from config, otherwise defer to set_attack_parameters
            try:
                if 'ATTACK_TYPE' in globals() and ATTACK_TYPE:
                    self.attack = Attack(ATTACK_TYPE)
                    print(f"Client {client_id}: Initialized as malicious with {ATTACK_TYPE} attack")
                else:
                    # Attack type will be set later via set_attack_parameters
                    self.attack = None
                    print(f"Client {client_id}: Initialized as malicious (attack type to be set)")
            except NameError:
                # ATTACK_TYPE not defined - attack will be set via set_attack_parameters
                self.attack = None
                print(f"Client {client_id}: Initialized as malicious (attack type to be set)")

    def compute_gradients(self, model, data_loader):
        """Compute gradients with improved numerical stability."""
        model.train()
        gradients = []
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            batch_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    # Don't normalize - preserve original gradient magnitude
                    grad = param.grad.detach().clone()
                    batch_gradients.append(grad)
            
            if batch_gradients:
                # Concatenate all gradients
                flat_gradients = torch.cat([g.flatten() for g in batch_gradients])
                
                # Apply gradient clipping for stability
                grad_norm = torch.norm(flat_gradients)
                if grad_norm > MAX_GRADIENT_NORM:
                    flat_gradients = flat_gradients * (MAX_GRADIENT_NORM / grad_norm)
                    
                gradients.append(flat_gradients)
            
            # Clear gradients
            model.zero_grad()
        
        if not gradients:
            return None
        
        # Average gradients across batches
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Only clip, don't normalize to unit norm
        grad_norm = torch.norm(avg_gradients)
        if grad_norm > MAX_GRADIENT_NORM:
            avg_gradients = avg_gradients * (MAX_GRADIENT_NORM / grad_norm)
            print(f"Average gradient clipped from {grad_norm:.4f} to {MAX_GRADIENT_NORM:.4f}")
        else:
            print(f"Average gradient norm: {grad_norm:.4f}")
        
        return avg_gradients

    def train(self, global_model, round_idx=0):
        """
        Train the local model on local data for local_epochs.
        
        Args:
            global_model: The global model to start from
            round_idx: The current round index
            
        Returns:
            tuple: (gradient, features) where gradient is the model update and features
                   is the gradient feature vector
        """
        # Copy the global model for local training
        self.model.load_state_dict(global_model.state_dict())
        
        # Perform client local training
        self.model.train()
        
        # Set optimizer and move to correct device
        # CRITICAL: Read LR from config at runtime
        current_lr = getattr(config, 'LR', LR)
        self.optimizer = optim.SGD(self.model.parameters(), lr=current_lr)
        
        # Use the instance local_epochs variable
        print(f"Client {self.client_id}: Training for {self.local_epochs} local epochs")
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Store batches that are too small to handle later
            small_batches = []
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Handle empty batches gracefully
                if len(data) == 0:
                    continue
                
                # Skip batches of size 1 that would cause BatchNorm issues - save for later
                if len(data) == 1:
                    small_batches.append((data, target))
                    continue
                    
                data, target = data.to(self.device), target.to(self.device)
                
                # Apply attack on data if this is a malicious client
                if self.is_malicious and hasattr(self, 'attack'):
                    if hasattr(self.attack, 'apply_attack'):
                        try:
                            data, target = self.attack.apply_attack(data, target)
                        except Exception as e:
                            print(f"Attack data modification failed: {str(e)}")
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                try:
                # Forward pass
                    output = self.model(data)
                    
                    # Calculate loss
                    loss = self.criterion(output, target)
                
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Track loss
                    epoch_loss += loss.item()
                    
                    # Track accuracy
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                except RuntimeError as e:
                    if "Expected more than 1 value per channel when training" in str(e):
                        print(f"BatchNorm error with batch size {len(data)}, skipping batch")
                        continue
                    else:
                        # Re-raise unexpected errors
                        raise e
                
                # Early stopping to prevent over-training by malicious clients
                if self.is_malicious and epoch >= min(self.local_epochs // 2, 1) and batch_idx >= len(self.train_loader) // 4:
                    print(f"Malicious client {self.client_id}: Early stopping training")
                    break
            
            # Try to process small batches by combining them if possible
            if small_batches:
                print(f"Processing {len(small_batches)} small batches by combining them")
                if len(small_batches) > 1:
                    # Combine small batches
                    combined_data = []
                    combined_target = []
                    for data, target in small_batches:
                        combined_data.append(data)
                        combined_target.append(target)
                    
                    data = torch.cat(combined_data, dim=0)
                    target = torch.cat(combined_target, dim=0)
                    
                    # Process the combined batch
                    data, target = data.to(self.device), target.to(self.device)
                    
                    try:
                        # Forward pass with combined batch
                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss.backward()
                        self.optimizer.step()
                        
                        # Track metrics
                        epoch_loss += loss.item()
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                    except RuntimeError as e:
                        print(f"Error processing combined small batches: {str(e)}")
            
            # Print epoch statistics
            accuracy = 100. * correct / total if total > 0 else 0
            avg_loss = epoch_loss / (batch_idx + 1) if batch_idx > 0 else 0
            print(f"Client {self.client_id}, Epoch {epoch+1}/{self.local_epochs}: Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")
        
        # Extract the learned model update
        # This is simply the difference from the global model
        client_gradient = self._compute_model_update(global_model, self.model)
        
        if client_gradient is None:
            print(f"Client {self.client_id}: Warning - gradient computation failed.")
            return None, None
            
        # Calculate gradient statistics
        client_grad_norm = torch.norm(client_gradient).item()
        client_grad_mean = torch.mean(client_gradient).item()
        client_grad_std = torch.std(client_gradient).item()
        
        print(f"\nRaw Gradient Statistics for Client {self.client_id}:")
        print(f"Norm: {client_grad_norm:.4f}")
        print(f"Mean: {client_grad_mean:.4f}")
        print(f"Std: {client_grad_std:.4f}")
        
        # Store original gradient before attack for malicious clients
        original_gradient = None
        if self.is_malicious:
            original_gradient = client_gradient.clone()
            original_norm = torch.norm(original_gradient).item()
            print(f"Client {self.client_id} (MALICIOUS): Original gradient norm before attack: {original_norm:.4f}")

        # Determine whether to apply the attack this round (dynamic attack schedule)
        _apply_attack_this_round = True
        if self.is_malicious and hasattr(self, 'dynamic_attack_schedule') and self.dynamic_attack_schedule:
            schedule = self.dynamic_attack_schedule
            if schedule == 'phase_transition':
                _transition_round = getattr(self, 'phase_transition_round', 12)
                _apply_attack_this_round = (round_idx >= _transition_round)
                phase_label = "ATTACK" if _apply_attack_this_round else "BENIGN"
                print(f"Client {self.client_id} (MALICIOUS): Round {round_idx + 1} → {phase_label} "
                      f"(transition at round {_transition_round + 1})")
            elif schedule == 'alternating':
                _apply_attack_this_round = (round_idx % 2 == 0)
                phase_label = "ATTACK" if _apply_attack_this_round else "BENIGN"
                print(f"Client {self.client_id} (MALICIOUS): Round {round_idx + 1} → {phase_label} (alternating)")

        # If this is a malicious client, apply the gradient attack (unless benign phase)
        if self.is_malicious and hasattr(self, 'attack') and _apply_attack_this_round:
            try:
                modified_gradient = self.attack.apply_gradient_attack(client_gradient)
                
                # Verify that the attack had an effect
                modified_norm = torch.norm(modified_gradient).item()
                grad_diff = torch.norm(modified_gradient - client_gradient).item()
                
                print(f"Client {self.client_id}: Applied {self.attack.attack_type}:")
                print(f"  Original gradient norm: {client_grad_norm:.4f}")
                print(f"  Modified gradient norm: {modified_norm:.4f}")
                # Avoid division by zero when client_grad_norm is 0
                if client_grad_norm > 1e-10:
                    pct_change = (modified_norm/client_grad_norm - 1)*100
                    print(f"  Norm increased by: {modified_norm - client_grad_norm:.4f} ({pct_change:.2f}%)")
                else:
                    print(f"  Norm increased by: {modified_norm - client_grad_norm:.4f} (original norm ~0)")
                print(f"  Gradient difference norm: {grad_diff:.4f}")
                
                # Save original gradient for server to analyze
                self.original_gradient = client_gradient.clone()
                
                # Update client gradient with attacked version
                client_gradient = modified_gradient
            except Exception as e:
                print(f"Client {self.client_id}: Failed to apply gradient attack: {str(e)}")
            
        # Calculate the feature representation for this gradient if needed for trust scoring
        gradient_features = self._extract_gradient_features(client_gradient)
        
        # Print the normalized gradient norm
        grad_norm = torch.norm(client_gradient).item()
        normalized_norm = min(grad_norm / MAX_GRADIENT_NORM, 1.0)  # Ensure it's capped at 1.0
        print(f"Client {self.client_id} gradient features:")
        print(f"  Raw gradient norm: {grad_norm:.4f}")
        print(f"  Normalized gradient norm: {normalized_norm:.4f}")
        if self.is_malicious and original_gradient is not None:
            original_norm = torch.norm(original_gradient).item()
            print(f"  Original norm (before attack): {original_norm:.4f}")
            print(f"  Normalized original norm: {min(original_norm / MAX_GRADIENT_NORM, 1.0):.4f}")
            # Avoid division by zero when original_norm is 0
            if original_norm > 1e-10:
                print(f"  Norm increase from attack: {grad_norm - original_norm:.4f} ({(grad_norm/original_norm - 1)*100:.2f}%)")
            else:
                print(f"  Norm increase from attack: {grad_norm - original_norm:.4f} (original norm ~0, cannot compute percentage)")
        
        return client_gradient, gradient_features

    def _compute_model_update(self, global_model, local_model):
        """Compute gradient update by parameter difference with improved handling for FedBN."""
        # Initialize gradient list and BatchNorm gradient list (for FedBN)
        grad_list = []
        bn_grad_list = []
        
        # Make sure models are on the same device
        global_model = global_model.to(self.device)
        local_model = local_model.to(self.device)
        
        # Check if we're using FedBN or FedBN+FedProx
        is_fedbn = AGGREGATION_METHOD in ['fedbn', 'fedbn_fedprox']
        
        # If using FedBN, identify BatchNorm layers
        bn_layers = set()
        if is_fedbn:
            for name, module in global_model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    bn_layers.add(name)
                    print(f"Client {self.client_id}: Identified BatchNorm layer: {name}")
            
            # Also check for common BatchNorm naming patterns
            for name, _ in global_model.named_parameters():
                if '.bn.' in name or 'downsample.1' in name or name.endswith('.bn.weight') or name.endswith('.bn.bias'):
                    parts = name.split('.')
                    bn_name = '.'.join(parts[:-1])
                    bn_layers.add(bn_name)
            
            print(f"Client {self.client_id}: Found {len(bn_layers)} BatchNorm layers")
        
        # Compute gradient based on parameter differences
        for (global_name, global_param), (local_name, local_param) in zip(
            global_model.named_parameters(), local_model.named_parameters()
        ):
            if global_name != local_name:
                print(f"Parameter name mismatch: {global_name} vs {local_name}")
                continue
            
            # Check if this is a BatchNorm parameter
            is_bn_param = False
            if is_fedbn:
                # Get module path without parameter name
                param_path = global_name.rsplit('.', 1)[0]
                if param_path in bn_layers or '.bn.' in global_name or 'downsample.1' in global_name:
                    is_bn_param = True
                    
            # Compute parameter difference (gradient)
            param_diff = local_param.data - global_param.data
            
            # For FedBN, store BatchNorm gradients separately
            if is_fedbn and is_bn_param:
                bn_grad_list.append(param_diff.view(-1))
                # Insert zeros as placeholders in the main gradient
                grad_list.append(torch.zeros_like(param_diff.view(-1)))
                if VERBOSE:
                    print(f"Client {self.client_id}: Preserving BatchNorm gradient for {global_name}")
                    print(f"  Gradient norm: {param_diff.norm().item():.4f}")
            else:
                grad_list.append(param_diff.view(-1))
        
        # Concatenate all gradients
        if not grad_list:
            return None
            
        gradient = torch.cat(grad_list)
        
        # Calculate gradient norm before any processing
        raw_norm = gradient.norm().item()
        print(f"Raw gradient norm before clipping: {raw_norm:.4f}")
        
        # If using FedBN, log stats about the BN parameters
        if is_fedbn and bn_grad_list:
            bn_gradient = torch.cat(bn_grad_list)
            bn_norm = bn_gradient.norm().item()
            print(f"Client {self.client_id}: BatchNorm gradient norm: {bn_norm:.4f}")
            print(f"Client {self.client_id}: BatchNorm parameters make up {bn_gradient.numel() / (gradient.numel() + bn_gradient.numel()) * 100:.2f}% of total parameters")
        
        # Apply adaptive clipping only if the gradient is abnormally large
        # This preserves the natural gradient magnitudes while preventing exploding gradients
        if raw_norm > MAX_GRADIENT_NORM * 2:  # Only clip if significantly larger than threshold
            scaling_factor = MAX_GRADIENT_NORM / raw_norm
            gradient = gradient * scaling_factor
            print(f"Clipped gradient norm from {raw_norm:.4f} to {gradient.norm().item():.4f}")
        
        # Print final gradient norm
        print(f"Client {self.client_id} gradient norm: {gradient.norm().item():.4f}")
        
        return gradient

    def _extract_gradient_features(self, gradient):
        """
        Compute feature vector for a gradient.
        
        Args:
            gradient: The gradient to analyze
            
        Returns:
            Tensor with feature values [feature_dim]
        """
        # Use 6 features if Shapley is enabled, otherwise 5
        feature_dim = 6 if ENABLE_SHAPLEY else 5
        features = torch.zeros(feature_dim, device=self.device)
        
        # 1. Set a placeholder for VAE reconstruction error (server will compute this)
        features[0] = 0.5  # Default value, will be computed by server
        
        # 2. Root similarity (placeholder, will be computed by server)
        features[1] = 0.5  # Default value
        
        # 3. Client similarity (placeholder, will be computed by server)
        features[2] = 0.5  # Default value
        
        # 4. Gradient norm (normalized with improved method)
        grad_norm = torch.norm(gradient).item()
        
        # Linear normalization relative to MAX_GRADIENT_NORM
        # Always cap at 1.0 to preserve scale
        max_norm = MAX_GRADIENT_NORM if 'MAX_GRADIENT_NORM' in globals() else 10.0
        norm_feature = min(grad_norm / max_norm, 1.0)  # Ensure it's capped at 1.0
        features[3] = torch.tensor(norm_feature, device=self.device)
        
        # 5. Set a placeholder for consistency (server will compute this)
        features[4] = 0.5  # Default value
        
        # 6. Shapley value (placeholder, will be computed by server)
        if ENABLE_SHAPLEY and feature_dim > 5:
            features[5] = 0.5  # Default value
        
        # Print feature values for debugging
        print(f"\nClient {self.client_id} gradient features:")
        print(f"  Raw gradient norm: {grad_norm:.4f}")
        print(f"  Normalized gradient norm: {features[3].item():.4f}")
        if self.is_malicious:
            print(f"  (This client is malicious)")
            if hasattr(self, 'original_gradient') and self.original_gradient is not None:
                orig_norm = torch.norm(self.original_gradient).item()
                print(f"  Original gradient norm (before attack): {orig_norm:.4f}")
                print(f"  Normalized original norm: {min(orig_norm / max_norm, 1.0):.4f}")
                # Avoid division by zero when orig_norm is 0
                if orig_norm > 1e-10:
                    print(f"  Norm increase from attack: {grad_norm - orig_norm:.4f} ({(grad_norm/orig_norm - 1)*100:.2f}%)")
                else:
                    print(f"  Norm increase from attack: {grad_norm - orig_norm:.4f} (original norm ~0)")
        
        return features

    def _apply_attack(self, gradient):
        """
        Apply attack to gradient if client is malicious.
        
        Args:
            gradient: Client gradient to modify
            
        Returns:
            Modified gradient if malicious, original otherwise
        """
        if not self.is_malicious:
            return gradient
        
        # Store original gradient for debugging and comparison
        self.original_gradient = gradient.clone().detach()
        
        # Print original gradient statistics
        original_norm = torch.norm(gradient).item()
        print(f"Original gradient (before attack) - Norm: {original_norm:.4f}")
        
        # Apply attack to gradient
        if hasattr(self, 'attack'):
            modified_gradient = self.attack.apply_gradient_attack(gradient)
            
            # Print attack statistics
            modified_norm = torch.norm(modified_gradient).item()
            # Avoid division by zero when original_norm is 0
            if original_norm > 1e-10:
                relative_change = (modified_norm / original_norm - 1) * 100
                print(f"Attack: {self.attack.attack_type} - Norm after attack: {modified_norm:.4f}")
                print(f"Relative change: {relative_change:.2f}%")
            else:
                print(f"Attack: {self.attack.attack_type} - Norm after attack: {modified_norm:.4f}")
                print(f"Relative change: N/A (original norm was ~0)")
            
            return modified_gradient
        else:
            print("Warning: Malicious client has no attack object. Using original gradient.")
            return gradient

    def _get_gradient(self, global_model):
        """Get gradient as difference between local and global model parameters with enhanced features."""
        gradient = []
        for local_param, global_param in zip(self.model.parameters(), global_model.parameters()):
            if local_param.grad is not None:
                diff = local_param.data - global_param.data
                gradient.append(diff.view(-1))
        
        if not gradient:
            return None
        
        gradient = torch.cat(gradient)
        
        # Calculate gradient statistics
        grad_norm = torch.norm(gradient).item()
        grad_mean = gradient.mean().item()
        grad_std = gradient.std().item()
        
        # Calculate higher-order statistics
        normalized_grad = (gradient - grad_mean) / (grad_std + 1e-8)
        grad_skew = torch.mean(normalized_grad ** 3).item()
        grad_kurtosis = torch.mean(normalized_grad ** 4).item()
        
        # Store statistics
        self.training_history['gradient_norms'].append(grad_norm)
        self.training_history['feature_stats']['mean'].append(grad_mean)
        self.training_history['feature_stats']['std'].append(grad_std)
        self.training_history['feature_stats']['skewness'].append(grad_skew)
        self.training_history['feature_stats']['kurtosis'].append(grad_kurtosis)
            
        # Note: We don't apply attack here because it's already applied in the train method
        print(f"\nRaw Gradient Statistics for Client {self.client_id}:")
        print(f"Norm: {grad_norm:.4f}")
        print(f"Mean: {grad_mean:.4f}")
        print(f"Std: {grad_std:.4f}")
        
        return gradient

    def _get_raw_gradient(self, global_model):
        """Get raw gradient without any modifications and with enhanced statistics."""
        gradient = []
        for local_param, global_param in zip(self.model.parameters(), global_model.parameters()):
            if local_param.grad is not None:
                diff = local_param.data - global_param.data
                gradient.append(diff.view(-1))
        
        if not gradient:
            return None
        
        gradient = torch.cat(gradient)
        
        # Calculate and log gradient statistics
        grad_norm = torch.norm(gradient).item()
        grad_mean = gradient.mean().item()
        grad_std = gradient.std().item()
        
        print(f"\nRaw Gradient Statistics for Client {self.client_id}:")
        print(f"Norm: {grad_norm:.4f}")
        print(f"Mean: {grad_mean:.4f}")
        print(f"Std: {grad_std:.4f}")
        
        return gradient

    def set_attack_parameters(self, attack_type=None, **kwargs):
        """
        Set attack parameters for a malicious client.
        
        Args:
            attack_type: Type of attack to perform
            **kwargs: Additional attack parameters such as scaling_factor, partial_percent
        """
        if not self.is_malicious:
            print(f"Client {self.client_id} is not malicious. Cannot set attack parameters.")
            return
            
        # Set attack type
        if attack_type is not None:
            self.attack = Attack(attack_type)
            print(f"Client {self.client_id}: Attack type set to {attack_type}")
            
        # Store additional attack parameters
        self.attack_params = kwargs
        
        # Print attack parameters if any
        if kwargs:
            for key, value in kwargs.items():
                print(f"Client {self.client_id}: Attack parameter {key} = {value}")
        
        # Set attack parameters directly on the Attack object
        if hasattr(self, 'attack') and self.attack_params:
            for key, value in self.attack_params.items():
                setattr(self.attack, key, value) 