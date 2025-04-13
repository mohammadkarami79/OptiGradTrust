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
        for i in range(NUM_CLIENTS):
            is_malicious = i in malicious_clients
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
        train_vae(self.vae, self.root_gradients)
        
        print("\n=== Training Dual Attention ===")
        features = []
        labels = []
        print("\nCollecting client features...")
        for i, client in enumerate(self.clients):
            print(f"\nProcessing client {i}...")
            grad, _ = client.train(self.global_model)
            grad_input = grad.unsqueeze(0).to(device)
            recon_batch, mu, logvar = self.vae(grad_input)
            RE_val = torch.nn.functional.mse_loss(recon_batch, grad_input, reduction='sum').item()
            
            cos_root_vals = [torch.nn.functional.cosine_similarity(grad, r, dim=0).item() for r in self.root_gradients]
            mean_cosine_root = np.mean(cos_root_vals)
            
            neighbor_sims = []
            for j, other_client in enumerate(self.clients):
                if i != j:
                    other_grad, _ = other_client.train(self.global_model)
                    sim = torch.nn.functional.cosine_similarity(grad, other_grad, dim=0).item()
                    neighbor_sims.append(sim)
            mean_neighbor_sim = np.mean(neighbor_sims)
            
            features.append([RE_val, mean_cosine_root, mean_neighbor_sim, 1.0])
            labels.append(1 if client.is_malicious else 0)
            print(f"Client {i} features: RE={RE_val:.4f}, CosRoot={mean_cosine_root:.4f}, NeighborSim={mean_neighbor_sim:.4f}")
        
        features = np.array(features)
        f0 = features[:, 0]
        f1 = features[:, 1]
        f2 = features[:, 2]
        f0_norm = (f0 - f0.min()) / (f0.max() - f0.min() + 1e-8)
        f1_norm = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-8)
        f2_norm = (f2 - f2.min()) / (f2.max() - f2.min() + 1e-8)
        features_normalized = np.stack((f0_norm, f1_norm, f2_norm, features[:, 3]), axis=1)
        
        features_tensor = torch.tensor(features_normalized, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        train_dual_attention(self.dual_attention, features_tensor, labels_tensor)

    def train(self):
        print("\n=== Starting Federated Training ===")
        for epoch in range(GLOBAL_EPOCHS):
            print(f"\nGlobal Epoch {epoch + 1}/{GLOBAL_EPOCHS}")
            client_gradients = []
            features = []
            
            print("\nCollecting client updates...")
            for i, client in enumerate(self.clients):
                print(f"\nProcessing client {i}...")
                grad, raw_grad = client.train(self.global_model)
                client_gradients.append(grad)
                
                grad_input = grad.unsqueeze(0).to(device)
                recon_batch, mu, logvar = self.vae(grad_input)
                RE_val = torch.nn.functional.mse_loss(recon_batch, grad_input, reduction='sum').item()
                
                cos_root_vals = [torch.nn.functional.cosine_similarity(grad, r, dim=0).item() for r in self.root_gradients]
                mean_cosine_root = np.mean(cos_root_vals)
                
                neighbor_sims = []
                for other_grad in client_gradients[:-1]:
                    sim = torch.nn.functional.cosine_similarity(grad, other_grad, dim=0).item()
                    neighbor_sims.append(sim)
                mean_neighbor_sim = np.mean(neighbor_sims) if neighbor_sims else 0
                
                features.append([RE_val, mean_cosine_root, mean_neighbor_sim, torch.norm(raw_grad).item()])
                print(f"Client {i} features: RE={RE_val:.4f}, CosRoot={mean_cosine_root:.4f}, NeighborSim={mean_neighbor_sim:.4f}, GradNorm={torch.norm(raw_grad).item():.4f}")
            
            features = np.array(features)
            f0 = features[:, 0]
            f1 = features[:, 1]
            f2 = features[:, 2]
            f0_norm = (f0 - f0.min()) / (f0.max() - f0.min() + 1e-8)
            f1_norm = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-8)
            f2_norm = (f2 - f2.min()) / (f2.max() - f2.min() + 1e-8)
            features_normalized = np.stack((f0_norm, f1_norm, f2_norm, features[:, 3]), axis=1)
            
            features_tensor = torch.tensor(features_normalized, dtype=torch.float32).to(device)
            with torch.no_grad():
                trust_scores = self.dual_attention(features_tensor, features_tensor.mean(dim=0, keepdim=True))
                trust_scores = torch.sigmoid(trust_scores)
                trust_scores = trust_scores.detach().cpu().numpy()
            
            print("\nTrust scores:")
            for i, score in enumerate(trust_scores):
                print(f"Client {i}: {score:.4f}")
            
            total_trust = sum(trust_scores)
            if total_trust == 0:
                total_trust += 1e-8
            normalized_trust_scores = trust_scores / total_trust
            
            print("\nNormalized trust scores:")
            for i, score in enumerate(normalized_trust_scores):
                print(f"Client {i}: {score:.4f}")
            
            aggregated_gradient = torch.zeros_like(client_gradients[0])
            for grad, weight in zip(client_gradients, normalized_trust_scores):
                aggregated_gradient += grad * weight
            
            index = 0
            new_global_weights = {}
            for name, param in self.global_model.state_dict().items():
                param_size = param.numel()
                param_shape = param.shape
                param_update = aggregated_gradient[index: index + param_size].view(param_shape)
                new_global_weights[name] = param + param_update
                index += param_size
            
            self.global_model.load_state_dict(new_global_weights)
            
            if epoch % 10 == 0 or epoch == GLOBAL_EPOCHS - 1:
                test_err = test(self.global_model, self.test_dataset)
                print(f"\nEpoch {epoch + 1}, Testing Error Rate: {test_err:.4f}")
        
        final_err = test(self.global_model, self.test_dataset)
        print(f"\n=== Training Completed ===")
        print(f"Final Testing Error Rate: {final_err:.4f}") 