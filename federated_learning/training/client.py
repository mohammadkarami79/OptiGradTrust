import torch
import torch.optim as optim
import copy
from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.attacks.attack_utils import simulate_attack
from federated_learning.training.training_utils import client_update

class Client:
    def __init__(self, client_id, dataset, is_malicious=False):
        """
        Initialize a client for federated learning.
        
        Args:
            client_id: Unique identifier for the client
            dataset: Training dataset for the client
            is_malicious: Whether this client is malicious or benign
        """
        self.client_id = client_id
        self.dataset = dataset
        self.is_malicious = is_malicious
        self.model = CNNMnist().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=LR)
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    def train(self, global_model, epochs=None):
        """
        Train the local model based on the global model.
        
        Args:
            global_model: The global model from the server
            epochs: Number of local training epochs (optional, defaults to LOCAL_EPOCHS_CLIENT)
            
        Returns:
            normalized_grad: Normalized gradient for aggregation
            raw_grad: Raw gradient before normalization
        """
        # By default, use LOCAL_EPOCHS_CLIENT (10) for regular federated training
        # Only use custom epochs if explicitly provided (for Dual Attention training)
        if epochs is None:
            epochs = LOCAL_EPOCHS_CLIENT
        
        # Initialize local model with global model
        self.model.load_state_dict(global_model.state_dict())
        
        # Train local model with local data
        state_dict = client_update(self.model, self.optimizer, self.train_loader, epochs)
        self.model.load_state_dict(state_dict)
        
        # Compute gradient (difference between local and global models)
        grad_list = []
        for (name, pg), (_, plg) in zip(global_model.named_parameters(), self.model.named_parameters()):
            diff = (plg.data - pg.data).view(-1)
            grad_list.append(diff)
        raw_grad = torch.cat(grad_list).detach()
        
        # Apply attack if this is a malicious client
        if self.is_malicious:
            raw_grad = simulate_attack(raw_grad, ATTACK_TYPE)
        
        # Normalize gradient for aggregation
        norm_val = torch.norm(raw_grad) + 1e-8
        normalized_grad = raw_grad / norm_val
        
        return normalized_grad, raw_grad 