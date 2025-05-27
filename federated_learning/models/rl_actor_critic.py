import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..config import config

class Actor(nn.Module):
    """
    Actor network that takes client feature vectors as input and outputs 
    gradient aggregation weights.
    """
    def __init__(self, input_dim=6, hidden_dims=None):
        """
        Initialize the Actor network.
        
        Args:
            input_dim: Dimension of each client's feature vector (default: 6)
            hidden_dims: List of hidden layer dimensions
        """
        super(Actor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.RL_ACTOR_HIDDEN_DIMS
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Final layer outputs a single scalar per client
        layers.append(nn.Linear(prev_dim, 1))
        
        # Build the MLP
        self.mlp = nn.Sequential(*layers)
        
        # Temperature parameter for softmax (can be annealed during training)
        self.register_buffer('temperature', torch.tensor(config.RL_INITIAL_TEMP))
    
    def forward(self, features, return_probs=True):
        """
        Forward pass through the actor network.
        
        Args:
            features: Client gradient features [batch_size, feature_dim]
            return_probs: Whether to return probabilities (weights) or logits
            
        Returns:
            weights or logits depending on return_probs parameter
        """
        # Input normalization
        x = self.input_norm(features)
        
        # Forward through MLP
        logits = self.mlp(x).squeeze(-1)
        
        if return_probs:
            # Apply softmax with temperature
            weights = F.softmax(logits / self.temperature, dim=0)
            return weights
        else:
            return logits
    
    def get_entropy(self, logits=None, features=None):
        """
        Calculate entropy of the policy for exploration bonus.
        Higher entropy means more exploration.
        
        Args:
            logits: Pre-computed logits, or None to compute from features
            features: Client features to compute logits if not provided
            
        Returns:
            entropy: Policy entropy value
        """
        if logits is None:
            if features is None:
                raise ValueError("Either logits or features must be provided")
            with torch.no_grad():
                # Get logits from features
                x = self.input_norm(features)
                logits = self.mlp(x).squeeze(-1)
        
        # Calculate probabilities
        probs = F.softmax(logits / self.temperature, dim=0)
        
        # Calculate entropy: -sum(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs)
        
        return entropy
    
    def set_temperature(self, temp):
        """Set the temperature parameter for softmax."""
        self.temperature = torch.tensor(temp, device=self.temperature.device)


class Critic(nn.Module):
    """
    Critic network that estimates the value (expected return) of the current state.
    The state is represented by the matrix of client feature vectors.
    """
    def __init__(self, input_dim=6, hidden_dims=None):
        """
        Initialize the Critic network.
        
        Args:
            input_dim: Dimension of each client's feature vector (default: 6)
            hidden_dims: List of hidden layer dimensions
        """
        super(Critic, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.RL_CRITIC_HIDDEN_DIMS
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Final layer outputs a single scalar value estimate
        layers.append(nn.Linear(prev_dim, 1))
        
        # Build the MLP
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Forward pass through the critic network.
        
        Args:
            features: Client gradient features [batch_size, feature_dim]
            
        Returns:
            value: Estimated state value
        """
        # Sort features by client_id if provided as (client_id, features) pairs
        if isinstance(features, tuple) and len(features) == 2:
            client_ids, feature_vectors = features
            # Sort by client_id
            sorted_indices = torch.argsort(client_ids)
            features = feature_vectors[sorted_indices]
        
        # Process each client's features
        normalized_features = self.input_norm(features)
        
        # Pass through the MLP
        per_client_values = self.mlp(normalized_features)
        
        # Average the values across clients to get the overall state value
        value = torch.mean(per_client_values)
        
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic model for RL-based gradient aggregation.
    """
    def __init__(self, input_dim=6):
        """
        Initialize the ActorCritic model.
        
        Args:
            input_dim: Dimension of each client's feature vector (default: 6)
        """
        super(ActorCritic, self).__init__()
        
        # Create actor and critic networks
        self.actor = Actor(input_dim=input_dim)
        self.critic = Critic(input_dim=input_dim)
        
        # Initialize saved log probs and rewards for policy gradient
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, features):
        """
        Forward pass through both actor and critic networks.
        
        Args:
            features: Client gradient features [batch_size, feature_dim]
            
        Returns:
            weights: Aggregation weights for each client
            value: Estimated state value
        """
        # Get weights from actor and value from critic
        weights = self.actor(features)
        value = self.critic(features)
        
        return weights, value
    
    def get_weights(self, features):
        """
        Get aggregation weights for clients based on their features.
        
        Args:
            features: Client gradient features [batch_size, feature_dim]
            
        Returns:
            weights: Aggregation weights for each client
        """
        with torch.no_grad():
            weights = self.actor(features)
        return weights
    
    def select_action(self, features):
        """
        Select an action (aggregation weights) and save the log probability.
        
        Args:
            features: Client gradient features [batch_size, feature_dim]
            
        Returns:
            weights: Aggregation weights for each client
        """
        # Get logits from actor
        logits = self.actor(features, return_probs=False)
        
        # Sample weights using softmax with temperature
        probs = F.softmax(logits / self.actor.temperature, dim=0)
        m = torch.distributions.Categorical(probs)
        selected_indices = m.sample()
        
        # Create one-hot encoded weights
        weights = torch.zeros_like(probs)
        weights[selected_indices] = 1.0
        
        # Save log probability for training
        self.saved_log_probs.append(m.log_prob(selected_indices))
        
        return weights 