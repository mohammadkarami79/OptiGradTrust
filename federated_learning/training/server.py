import torch
import torch.optim as optim
import numpy as np
import random
import copy
from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.models.vae import GradientVAE
from federated_learning.models.attention import DualAttention
from federated_learning.training.training_utils import train_vae, train_dual_attention, test
from federated_learning.training.client import Client
from federated_learning.data.dataset import (
    LabelFlippingDataset, BackdoorDataset, AdaptiveAttackDataset,
    MinMaxAttackDataset, MinSumAttackDataset, AlternatingAttackDataset, 
    TargetedAttackDataset, GradientInversionAttackDataset
)

class Server:
    def __init__(self, root_dataset, client_datasets, test_dataset):
        print("\n=== Initializing Server ===")
        print("Creating global model...")
        self.global_model = CNNMnist().to(device)
        self.test_dataset = test_dataset
        self.root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=len(root_dataset), shuffle=True)
        
        print("\nInitializing clients...")
        self.clients = []
        malicious_clients = random.sample(range(NUM_CLIENTS), NUM_MALICIOUS)
        print("\n=== Malicious Clients ===")
        print(f"Malicious clients: {malicious_clients}")
        print("=== Client Information ===")
        
        # Get number of classes from the dataset
        self.num_classes = 10  # Default for MNIST
        
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
                
                self.clients.append(Client(i, wrapped_dataset, is_malicious))
            else:
                self.clients.append(Client(i, client_datasets[i], is_malicious))
                
            print(f"Client {i}: {'MALICIOUS' if is_malicious else 'HONEST'}, Dataset size: {len(client_datasets[i])}")
        
        print("\nInitializing VAE and Dual Attention...")
        self.vae = GradientVAE(input_size=sum(p.numel() for p in self.global_model.parameters())).to(device)
        self.dual_attention = DualAttention(feature_size=4).to(device)
        
        print("\nCollecting root gradients...")
        self.root_gradients = self._collect_root_gradients()
        print(f"Collected {len(self.root_gradients)} root gradients")
        
        print("\nTraining VAE and Dual Attention...")
        self._train_models()

    def _collect_root_gradients(self):
        print("\n=== Collecting Root Gradients ===")
        root_gradients = []
        root_model = copy.deepcopy(self.global_model)
        root_optimizer = optim.SGD(root_model.parameters(), lr=LR)
        criterion = torch.nn.NLLLoss()
        
        for epoch in range(LOCAL_EPOCHS_ROOT):
            epoch_loss = 0
            for data, target in self.root_loader:
                data = data.to(device)
                target = target.to(device)
                root_optimizer.zero_grad()
                output = root_model(data)
                loss = criterion(output, target)
                loss.backward()
                
                grad_vector = []
                for param in root_model.parameters():
                    grad_vector.append(param.grad.view(-1))
                grad_vector = torch.cat(grad_vector).detach()
                grad_vector = grad_vector / (torch.norm(grad_vector) + 1e-8)
                root_gradients.append(grad_vector)
                
                epoch_loss += loss.item()
                root_optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Root Training Epoch {epoch + 1}/{LOCAL_EPOCHS_ROOT}, Loss: {epoch_loss/len(self.root_loader):.4f}")
        
        return root_gradients

    def _train_models(self):
        print("\n=== Training VAE ===")
        # First, collect trusted gradients from root and benign clients
        all_trusted_gradients = []
        
        # 1. Add root gradients (already collected in self.root_gradients)
        print(f"Adding {len(self.root_gradients)} root gradients to VAE training set")
        all_trusted_gradients.extend(self.root_gradients)
        
        # 2. Collect gradients from benign clients (8 clients, 10 epochs each = 80 gradients)
        benign_clients = [client for client in self.clients if not client.is_malicious]
        print(f"\nCollecting {len(benign_clients)} benign client gradients...")
        
        benign_gradients = []
        for i, client in enumerate(benign_clients):
            print(f"\nProcessing benign client {client.client_id}...")
            for epoch in range(LOCAL_EPOCHS_CLIENT):
                print(f"Epoch {epoch+1}/{LOCAL_EPOCHS_CLIENT}")
                grad, _ = client.train(self.global_model)
                normalized_grad = grad / (torch.norm(grad) + 1e-8)
                benign_gradients.append(normalized_grad)
                all_trusted_gradients.append(normalized_grad)
        
        print(f"\nTotal trusted gradients for VAE training: {len(all_trusted_gradients)}")
        print(f"- {len(self.root_gradients)} gradients from root")
        print(f"- {len(benign_gradients)} gradients from benign clients")
        
        # Train VAE on all trusted gradients (200 root + 80 benign = 280 total)
        train_vae(self.vae, all_trusted_gradients)
        
        print("\n=== Preparing Data for Dual Attention Training ===")
        features = []
        labels = []
        
        # 1. Process benign clients (8 clients, BENIGN_DA_EPOCHS epochs each)
        print(f"\nCollecting benign client features for Dual Attention (using {BENIGN_DA_EPOCHS} epochs per client)...")
        for i, client in enumerate(benign_clients):
            print(f"\nProcessing benign client {client.client_id} for Dual Attention...")
            client_features = []
            
            # Collect BENIGN_DA_EPOCHS gradients for each benign client
            for epoch in range(BENIGN_DA_EPOCHS):
                print(f"Epoch {epoch+1}/{BENIGN_DA_EPOCHS}")
                grad, raw_grad = client.train(self.global_model, epochs=LOCAL_EPOCHS_CLIENT)
                
                # Compute features for this gradient
                feature = self._compute_gradient_features(grad, raw_grad)
                client_features.append(feature)
                
                # Add to training data with label 0 (benign)
                features.append(feature)
                labels.append(0)
                
            print(f"Collected {len(client_features)} feature vectors from benign client {client.client_id}")
        
        # 2. Add features from root gradients (200 gradients)
        print("\nAdding features from root gradients...")
        root_features = []
        for i, root_grad in enumerate(self.root_gradients):
            # Compute features for root gradient
            root_feature = self._compute_gradient_features(root_grad, root_grad)
            root_features.append(root_feature)
            
            # Add to training data with label 0 (benign)
            features.append(root_feature)
            labels.append(0)
        
        # 3. Process malicious clients (2 clients, MALICIOUS_EPOCHS each, 10 attack types = 800 gradients)
        print(f"\nCollecting malicious client features for Dual Attention (using {MALICIOUS_EPOCHS} epochs per client)...")
        malicious_clients = [client for client in self.clients if client.is_malicious]
        
        # All attack types
        attack_types = ['label_flipping', 'scaling_attack', 'partial_scaling_attack', 
                       'backdoor_attack', 'adaptive_attack', 'min_max_attack', 
                       'min_sum_attack', 'alternating_attack', 'targeted_attack', 
                       'gradient_inversion_attack']
        
        for i, client in enumerate(malicious_clients):
            print(f"\nProcessing malicious client {client.client_id} for Dual Attention...")
            
            # Collect MALICIOUS_EPOCHS gradients for each malicious client
            client_raw_grads = []
            for epoch in range(MALICIOUS_EPOCHS):
                print(f"Epoch {epoch+1}/{MALICIOUS_EPOCHS}")
                # Get gradient without attack (for baseline)
                _, raw_grad = client.train(self.global_model, epochs=LOCAL_EPOCHS_CLIENT)
                client_raw_grads.append(raw_grad)
            
            # Now apply each attack type to each gradient
            for j, raw_grad in enumerate(client_raw_grads):
                for attack_type in attack_types:
                    # Import on-demand to avoid circular import
                    from federated_learning.attacks.attack_utils import simulate_attack
                    
                    # Apply attack to get malicious gradient
                    malicious_grad = simulate_attack(raw_grad, attack_type)
                    
                    # Normalize gradient
                    norm_val = torch.norm(malicious_grad) + 1e-8
                    normalized_grad = malicious_grad / norm_val
                    
                    # Compute features
                    feature = self._compute_gradient_features(normalized_grad, malicious_grad)
                    
                    # Add to training data with label 1 (malicious)
                    features.append(feature)
                    labels.append(1)
                    
                    if j == 0 and attack_type == attack_types[0]:
                        print(f"Sample malicious features: {feature}")
        
        print("\n=== Feature Dataset Summary ===")
        benign_count = sum(1 for label in labels if label == 0)
        malicious_count = sum(1 for label in labels if label == 1)
        print(f"Total feature vectors: {len(features)}")
        print(f"- Benign features (label 0): {benign_count}")
        print(f"- Malicious features (label 1): {malicious_count}")
        
        # Convert to numpy arrays and normalize
        features = np.array(features)
        labels = np.array(labels)
        
        # Normalize all features to [0,1] range
        features_normalized = np.zeros_like(features)
        for i in range(features.shape[1]):
            f_min = features[:, i].min()
            f_max = features[:, i].max()
            if f_max - f_min > 1e-8:  # Avoid division by zero
                features_normalized[:, i] = (features[:, i] - f_min) / (f_max - f_min)
            else:
                features_normalized[:, i] = 0.5  # Default value if all values are the same
        
        features_tensor = torch.tensor(features_normalized, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        print("\n=== Training Dual Attention ===")
        train_dual_attention(self.dual_attention, features_tensor, labels_tensor)
        
    def _compute_gradient_features(self, grad, raw_grad):
        """
        Compute feature vector for a gradient.
        
        Args:
            grad: Normalized gradient
            raw_grad: Raw gradient before normalization
            
        Returns:
            List of features [RE_val, mean_cosine_root, mean_neighbor_sim, grad_norm]
        """
        # Compute reconstruction error using VAE
        grad_input = grad.unsqueeze(0).to(device)
        recon_batch, mu, logvar = self.vae(grad_input)
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
        
        return [RE_val, mean_cosine_root, mean_neighbor_sim, grad_norm]

    def train(self):
        print("\n=== Starting Federated Training ===")
        for epoch in range(GLOBAL_EPOCHS):
            print(f"\nGlobal Epoch {epoch + 1}/{GLOBAL_EPOCHS}")
            client_gradients = []
            raw_gradients = []
            features = []
            
            print("\nCollecting client updates...")
            # First, collect all gradients
            for i, client in enumerate(self.clients):
                print(f"\nProcessing client {i}...")
                # Always use LOCAL_EPOCHS_CLIENT (10) for all clients in federated training
                grad, raw_grad = client.train(self.global_model, epochs=LOCAL_EPOCHS_CLIENT)
                client_gradients.append(grad)
                raw_gradients.append(raw_grad)
            
            # Now compute features with proper neighbor similarities
            for i, (grad, raw_grad) in enumerate(zip(client_gradients, raw_gradients)):
                print(f"\nComputing features for client {i}...")
                
                # Get basic features
                grad_input = grad.unsqueeze(0).to(device)
                recon_batch, mu, logvar = self.vae(grad_input)
                RE_val = torch.nn.functional.mse_loss(recon_batch, grad_input, reduction='sum').item()
                
                cos_root_vals = [torch.nn.functional.cosine_similarity(grad, r, dim=0).item() for r in self.root_gradients]
                mean_cosine_root = np.mean(cos_root_vals)
                
                # Now we can compute proper mean_neighbor_sim since we have all gradients
                neighbor_sims = []
                for j, other_grad in enumerate(client_gradients):
                    if i != j:  # Skip self-comparison
                        sim = torch.nn.functional.cosine_similarity(grad, other_grad, dim=0).item()
                        neighbor_sims.append(sim)
                
                mean_neighbor_sim = np.mean(neighbor_sims) if neighbor_sims else 0.0
                
                # Compute gradient norm
                grad_norm = torch.norm(raw_grad).item()
                
                # Create feature vector
                feature = [RE_val, mean_cosine_root, mean_neighbor_sim, grad_norm]
                features.append(feature)
                
                print(f"Client {i} features: RE={RE_val:.4f}, CosRoot={mean_cosine_root:.4f}, NeighborSim={mean_neighbor_sim:.4f}, GradNorm={grad_norm:.4f}")
            
            features = np.array(features)
            
            # Normalize features to [0,1] range for consistent evaluation
            features_normalized = np.zeros_like(features)
            for i in range(features.shape[1]):
                f_min = features[:, i].min()
                f_max = features[:, i].max()
                if f_max - f_min > 1e-8:  # Avoid division by zero
                    features_normalized[:, i] = (features[:, i] - f_min) / (f_max - f_min)
                else:
                    features_normalized[:, i] = 0.5  # Default value if all values are the same
            
            # Get trust scores using the trained dual attention model
            features_tensor = torch.tensor(features_normalized, dtype=torch.float32).to(device)
            with torch.no_grad():
                trust_scores = self.dual_attention(features_tensor, features_tensor.mean(dim=0, keepdim=True))
                trust_scores = torch.sigmoid(trust_scores)
                trust_scores = trust_scores.detach().cpu().numpy()
            
            print("\nTrust scores:")
            for i, score in enumerate(trust_scores):
                client_type = "MALICIOUS" if self.clients[i].is_malicious else "HONEST"
                print(f"Client {i} ({client_type}): {score:.4f}")
            
            # Ensure trust scores sum to non-zero value
            total_trust = sum(trust_scores)
            if total_trust == 0:
                total_trust += 1e-8
            
            # Normalize trust scores to create aggregation weights
            normalized_trust_scores = trust_scores / total_trust
            
            print("\nNormalized trust scores (aggregation weights):")
            for i, score in enumerate(normalized_trust_scores):
                print(f"Client {i}: {score:.4f}")
            
            # Calculate detection metrics
            malicious_detected = sum(1 for i, score in enumerate(trust_scores) 
                                  if self.clients[i].is_malicious and score < 0.5)
            honest_passed = sum(1 for i, score in enumerate(trust_scores) 
                             if not self.clients[i].is_malicious and score >= 0.5)
            total_malicious = sum(1 for client in self.clients if client.is_malicious)
            total_honest = sum(1 for client in self.clients if not client.is_malicious)
            
            # Calculate detection rates
            malicious_detection_rate = malicious_detected / total_malicious if total_malicious > 0 else 1.0
            false_positive_rate = 1 - (honest_passed / total_honest) if total_honest > 0 else 0.0
            
            print(f"\nMalicious Detection Rate: {malicious_detection_rate:.4f}")
            print(f"False Positive Rate: {false_positive_rate:.4f}")
            
            # Aggregate gradients using trust scores as weights
            aggregated_gradient = torch.zeros_like(client_gradients[0])
            for grad, weight in zip(client_gradients, normalized_trust_scores):
                aggregated_gradient += grad * weight
            
            # Update global model parameters
            index = 0
            new_global_weights = {}
            for name, param in self.global_model.state_dict().items():
                param_size = param.numel()
                param_shape = param.shape
                param_update = aggregated_gradient[index: index + param_size].view(param_shape)
                new_global_weights[name] = param + param_update
                index += param_size
            
            self.global_model.load_state_dict(new_global_weights)
            
            # Evaluate model periodically
            if epoch % 10 == 0 or epoch == GLOBAL_EPOCHS - 1:
                test_err = test(self.global_model, self.test_dataset)
                print(f"\nEpoch {epoch + 1}, Testing Error Rate: {test_err:.4f}")
        
        # Final evaluation
        final_err = test(self.global_model, self.test_dataset)
        print(f"\n=== Training Completed ===")
        print(f"Final Testing Error Rate: {final_err:.4f}") 