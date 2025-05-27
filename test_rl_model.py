import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

# Import just what we need
from federated_learning.models.rl_actor_critic import ActorCritic

def test_simple_features():
    """Test the RL actor-critic model with simple feature vectors."""
    print("Testing RL actor-critic model with simple feature vectors")
    
    # Create model
    feature_dim = 6
    actor_critic = ActorCritic(input_dim=feature_dim)
    
    # Create some sample feature vectors
    # Features in order: VAE recon error, root similarity, client similarity, norm, sign consistency, shapley
    # Honest client features (low recon error, high similarity, normal norm, high consistency)
    honest_features = torch.tensor([
        [0.2, 0.8, 0.7, 0.5, 0.9, 0.8],  # Honest client 1
        [0.3, 0.7, 0.8, 0.4, 0.8, 0.7],  # Honest client 2
        [0.1, 0.9, 0.9, 0.6, 0.9, 0.9],  # Honest client 3
    ])
    
    # Malicious client features (high recon error, low similarity, abnormal norm, low consistency)
    malicious_features = torch.tensor([
        [0.8, 0.3, 0.2, 0.9, 0.2, 0.2],  # Malicious client 1
        [0.7, 0.2, 0.3, 0.1, 0.3, 0.3],  # Malicious client 2
    ])
    
    # Combine all features
    all_features = torch.cat([honest_features, malicious_features], dim=0)
    
    # Get weights from actor
    with torch.no_grad():
        weights = actor_critic.actor(all_features)
    
    # Print weights
    print("\nInitial weights (should be roughly equal for all clients):")
    for i, weight in enumerate(weights):
        client_type = "Honest" if i < len(honest_features) else "Malicious"
        print(f"Client {i} ({client_type}): {weight.item():.4f}")
    
    # Simulate training with rewards
    print("\nSimulating 50 training iterations with rewards")
    
    # Create optimizers
    actor_optimizer = optim.Adam(actor_critic.actor.parameters(), lr=0.01)
    critic_optimizer = optim.Adam(actor_critic.critic.parameters(), lr=0.01)
    
    # Train for more iterations
    for i in range(50):
        # Get weights
        weights = actor_critic.select_action(all_features)
        
        # Simulate reward calculation (higher reward for honest-weighted gradients)
        honest_weight_sum = weights[:len(honest_features)].sum().item()
        malicious_weight_sum = weights[len(honest_features):].sum().item()
        
        # Reward based on how much weight is given to honest clients
        reward = 2.0 * honest_weight_sum - 1.0 * malicious_weight_sum
        
        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: Reward = {reward:.4f}, Honest weights = {honest_weight_sum:.4f}, Malicious weights = {malicious_weight_sum:.4f}")
        
        # Add reward to list
        actor_critic.rewards.append(reward)
        
        # Update policy every few iterations
        if (i + 1) % 5 == 0:
            # Calculate returns
            returns = []
            R = 0
            gamma = 0.99
            
            # Calculate returns in reverse order
            for r in actor_critic.rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            
            # Convert to tensor
            returns = torch.tensor(returns)
            
            # Normalize returns for stability
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
            # Actor loss
            policy_loss = []
            for log_prob, R in zip(actor_critic.saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            if policy_loss:
                policy_loss = torch.stack(policy_loss).sum()
                
                # Reset gradients
                actor_optimizer.zero_grad()
                
                # Backward pass
                policy_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(actor_critic.actor.parameters(), 1.0)
                
                # Update actor
                actor_optimizer.step()
            
            # Clear saved rewards and log probs
            actor_critic.rewards = []
            actor_critic.saved_log_probs = []
    
    # Get final weights
    with torch.no_grad():
        final_weights = actor_critic.actor(all_features)
    
    # Print final weights
    print("\nFinal weights (should prefer honest clients):")
    for i, weight in enumerate(final_weights):
        client_type = "Honest" if i < len(honest_features) else "Malicious"
        print(f"Client {i} ({client_type}): {weight.item():.4f}")

if __name__ == "__main__":
    test_simple_features() 