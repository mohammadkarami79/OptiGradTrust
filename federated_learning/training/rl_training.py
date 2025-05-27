import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
from collections import deque
from torch.utils.data import DataLoader, SubsetRandomSampler
from ..models.rl_actor_critic import ActorCritic
from ..config import config
from ..utils.gradient_features import compute_gradient_features as extract_features
from ..utils.data_utils import get_dataset
from ..models.cnn import CNNMnist
from federated_learning.training.training_utils import test


class RLAggregationEnv:
    """
    Reinforcement Learning environment for federated learning gradient aggregation.
    """
    def __init__(self, global_model, validation_loader, device=config.device):
        """
        Initialize the RL environment.
        
        Args:
            global_model: The global model being trained in federated learning
            validation_loader: DataLoader for validation data
            device: Device to use for computation
        """
        self.global_model = global_model
        self.validation_loader = validation_loader
        self.device = device
        
        # Keep track of validation performance
        self.best_validation_loss = float('inf')
        self.best_validation_acc = 0.0
        
        # Track dual attention baseline performance for comparison
        self.dual_attention_loss = float('inf')
        self.dual_attention_acc = 0.0
        
        # Cache for evaluation results to avoid redundant computations
        self.eval_cache = {}
        
        # Store initial model state
        self.save_model_state()
    
    def save_model_state(self):
        """Save current model state to restore it later."""
        self.model_state = {
            k: v.clone().detach() for k, v in self.global_model.state_dict().items()
        }
    
    def restore_model_state(self):
        """Restore model to saved state."""
        self.global_model.load_state_dict(self.model_state)
    
    def set_dual_attention_baseline(self, loss, acc):
        """
        Set the dual attention baseline performance.
        
        Args:
            loss: Validation loss achieved by dual attention
            acc: Validation accuracy achieved by dual attention
        """
        self.dual_attention_loss = loss
        self.dual_attention_acc = acc
        
        # Also update best metrics if baseline is better
        self.best_validation_loss = min(self.best_validation_loss, loss)
        self.best_validation_acc = max(self.best_validation_acc, acc)
    
    def apply_gradients(self, gradients, weights):
        """
        Apply weighted gradients to the global model.
        
        Args:
            gradients: List of gradients from clients
            weights: Tensor of weights for each client's gradient
            
        Returns:
            None
        """
        self.save_model_state()  # Save state before applying gradients
        
        # Get state dict to update
        state_dict = self.global_model.state_dict()
        
        # Track parameters with batch norm statistics to handle them separately
        bn_params = []
        for name, param in self.global_model.named_parameters():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                bn_params.append(name)
        
        # Apply weighted aggregation to each parameter
        for name, param in self.global_model.named_parameters():
            if name not in bn_params and param.requires_grad:
                # Initialize weighted gradient
                weighted_grad = torch.zeros_like(param.data)
                
                # Aggregate weighted gradients
                for i, client_grad in enumerate(gradients):
                    if name in client_grad:
                        weighted_grad += weights[i] * client_grad[name]
                
                # Apply gradient to parameter
                param.data -= weighted_grad
    
    def evaluate(self, use_subset=True):
        """
        Evaluate the global model on the validation set.
        
        Args:
            use_subset: If True, use a subset of validation data for faster evaluation
            
        Returns:
            loss: Validation loss
            accuracy: Validation accuracy
        """
        # Check if we have this evaluation result cached
        model_hash = hash(str(self.global_model.state_dict()))
        if model_hash in self.eval_cache:
            return self.eval_cache[model_hash]
        
        # If using subset, create a subset data loader
        if use_subset and hasattr(config, 'RL_VALIDATION_MINIBATCH') and config.RL_VALIDATION_MINIBATCH < 1.0:
            # Get total number of samples
            total_samples = len(self.validation_loader.dataset)
            
            # Calculate subset size
            subset_size = int(total_samples * config.RL_VALIDATION_MINIBATCH)
            
            # Create indices for subset
            indices = torch.randperm(total_samples)[:subset_size]
            
            # Create sampler and data loader
            sampler = SubsetRandomSampler(indices)
            subset_loader = DataLoader(
                self.validation_loader.dataset,
                batch_size=self.validation_loader.batch_size,
                sampler=sampler
            )
            
            # Evaluate on subset
            val_acc, val_loss = test(self.global_model, subset_loader)
        else:
            # Evaluate on full validation set
            val_acc, val_loss = test(self.global_model, self.validation_loader)
        
        # Cache the result
        self.eval_cache[model_hash] = (val_loss, val_acc)
        
        # Limit cache size to prevent memory issues
        if len(self.eval_cache) > 100:  # Keep only the 100 most recent evaluations
            oldest_key = next(iter(self.eval_cache))
            self.eval_cache.pop(oldest_key)
        
        return val_loss, val_acc
    
    def step(self, gradients, weights, compare_to_baseline=True):
        """
        Apply weighted gradients and evaluate the resulting model.
        
        Args:
            gradients: List of gradients from clients
            weights: Tensor of weights for each client's gradient
            compare_to_baseline: If True, reward is based on improvement over dual attention baseline
            
        Returns:
            reward: Reward signal based on validation performance
            done: Boolean indicating if episode is done
            info: Dictionary containing additional information
        """
        # Apply weighted gradients
        self.apply_gradients(gradients, weights)
        
        # Evaluate new model
        val_loss, val_acc = self.evaluate()
        
        # Calculate reward based on improvement in validation metrics
        if compare_to_baseline and hasattr(self, 'dual_attention_loss'):
            # Compare to dual attention baseline instead of best seen
            loss_improvement = self.dual_attention_loss - val_loss
            acc_improvement = val_acc - self.dual_attention_acc
            
            # Reward based on improvement over baseline
            loss_reward = loss_improvement * 10.0
            acc_reward = acc_improvement * 20.0
            
            # Combine rewards
            reward = loss_reward + acc_reward
            
            # Add bonus for achieving new best
            if val_loss < self.best_validation_loss:
                reward += 5.0
                self.best_validation_loss = val_loss
            
            if val_acc > self.best_validation_acc:
                reward += 5.0
                self.best_validation_acc = val_acc
        else:
            # Standard reward based on improvement over best seen
            if val_loss < self.best_validation_loss:
                # Improved loss - positive reward
                reward = (self.best_validation_loss - val_loss) * 10.0  # Scale reward
                self.best_validation_loss = val_loss
            else:
                # Worse loss - negative reward
                loss_degradation = val_loss - self.best_validation_loss
                
                # If loss degrades too much, strongly penalize
                if hasattr(config, 'RL_FALLBACK_THRESHOLD') and loss_degradation > config.RL_FALLBACK_THRESHOLD:
                    reward = -10.0
                    # Restore previous model state
                    self.restore_model_state()
                else:
                    reward = -loss_degradation * 5.0  # Smaller penalty for slight degradation
            
            # Bonus for accuracy improvement
            if val_acc > self.best_validation_acc:
                reward += (val_acc - self.best_validation_acc) * 20.0  # Bonus for accuracy
                self.best_validation_acc = val_acc
        
        # Create info dictionary
        info = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_loss': self.best_validation_loss,
            'best_val_acc': self.best_validation_acc
        }
        
        if hasattr(self, 'dual_attention_loss'):
            info['dual_attention_loss'] = self.dual_attention_loss
            info['dual_attention_acc'] = self.dual_attention_acc
            info['loss_vs_baseline'] = self.dual_attention_loss - val_loss
            info['acc_vs_baseline'] = val_acc - self.dual_attention_acc
        
        # Done is always False in FL since it's an ongoing process
        done = False
        
        return reward, done, info


def pretrain_rl_model(actor_critic, federated_data, device=config.device):
    """
    Pre-train the RL model using simulated federated learning rounds with various attack types.
    
    Args:
        actor_critic: ActorCritic model
        federated_data: FederatedDataset instance or similar
        device: Device to use for computation
        
    Returns:
        actor_critic: Trained ActorCritic model
    """
    print("Pre-training RL model...")
    
    # Check if we should skip pretraining
    if hasattr(config, 'RL_SKIP_PRETRAINING') and config.RL_SKIP_PRETRAINING:
        print("Skipping RL pre-training as RL_SKIP_PRETRAINING is True")
        return actor_critic
        
    # Set a reasonable default for pretraining episodes
    episodes = getattr(config, 'RL_PRETRAINING_EPISODES', 5)
    # For quick testing, use a very small number
    if episodes > 20:
        print(f"Reducing RL_PRETRAINING_EPISODES from {episodes} to 5 to speed up training")
        episodes = 5
    
    # Set a time limit for the entire pretraining
    MAX_TRAINING_TIME = 120  # 2 minutes max for pretraining
    start_time = time.time()
    
    # Initialize storage for features during training
    if not hasattr(actor_critic, 'saved_features'):
        actor_critic.saved_features = []
    
    # Set up optimizers
    actor_optimizer = optim.Adam(
        actor_critic.actor.parameters(), lr=config.RL_LEARNING_RATE
    )
    critic_optimizer = optim.Adam(
        actor_critic.critic.parameters(), lr=config.RL_LEARNING_RATE
    )
    
    # Set up environment
    # Create a model to use for simulation
    
    # Get dataset
    try:
        train_dataset, test_dataset = get_dataset(config.DATASET)
        
        # Create test loader
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing for stability
            pin_memory=False  # Avoid pin_memory for stability
        )
        
        # Create model
        if config.MODEL == 'CNN':
            if config.DATASET == 'MNIST':
                model = CNNMnist()
            elif config.DATASET == 'CIFAR10':
                # TODO: Add CIFAR10 model if available
                raise ValueError('CNN model for CIFAR10 not implemented')
            else:  # Alzheimer
                # TODO: Add Alzheimer model if available
                raise ValueError('CNN model for Alzheimer not implemented')
        else:
            raise ValueError(f"Unsupported model: {config.MODEL}")
        
        # Move model to device
        model = model.to(device)
        
        # Create environment
        env = RLAggregationEnv(model, test_loader, device)
    except Exception as e:
        print(f"Error setting up RL environment: {str(e)}")
        print("Skipping RL pre-training due to setup error")
        import traceback
        traceback.print_exc()
        return actor_critic
    
    # Define attack types for simulation
    attack_types = [
        'scaling_attack',
        'partial_scaling_attack',
        'sign_flipping_attack',
        'noise_attack',
        'label_flipping'
    ]
    
    # Training loop
    running_reward = 0
    
    # Lists to store trajectories
    actor_critic.saved_log_probs = []
    actor_critic.rewards = []
    
    # Schedule for temperature annealing
    initial_temp = config.RL_INITIAL_TEMP if hasattr(config, 'RL_INITIAL_TEMP') else 5.0
    min_temp = config.RL_MIN_TEMP if hasattr(config, 'RL_MIN_TEMP') else 0.5
    actor_critic.actor.set_temperature(initial_temp)
    
    # Track successful episodes
    successful_episodes = 0
    episode_errors = 0
    
    try:
        for episode in range(episodes):
            # Check time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_TRAINING_TIME:
                print(f"Time limit reached ({MAX_TRAINING_TIME}s). Stopping RL pre-training.")
                break
                
            # Reset environment (restore model to initial state)
            env.restore_model_state()
            
            # Select attack type for this episode
            # Cycle through attack types to ensure coverage
            attack_type = attack_types[episode % len(attack_types)]
            
            # Simulate clients and gradients
            num_clients = config.NUM_CLIENTS
            num_malicious = config.NUM_MALICIOUS
            
            # Generate honest gradients (normal gradients)
            honest_gradients = []
            for _ in range(num_clients - num_malicious):
                client_grads = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # Create normal gradient with appropriate shape
                        client_grads[name] = torch.randn_like(param.data) * 0.01
                honest_gradients.append(client_grads)
            
            # Generate malicious gradients based on attack type
            malicious_gradients = []
            for _ in range(num_malicious):
                client_grads = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # Create base gradient
                        base_grad = torch.randn_like(param.data) * 0.01
                        
                        # Apply attack based on type
                        if attack_type == 'scaling_attack':
                            # Scale the gradient by a large factor
                            scale_factor = getattr(config, 'SCALING_FACTOR', 20.0)
                            client_grads[name] = base_grad * scale_factor
                        
                        elif attack_type == 'partial_scaling_attack':
                            # Scale a portion of the gradient
                            percent = getattr(config, 'PARTIAL_SCALING_PERCENT', 0.4)
                            scale_factor = getattr(config, 'SCALING_FACTOR', 20.0)
                            mask = torch.rand_like(base_grad) < percent
                            scaled_grad = torch.where(mask, base_grad * scale_factor, base_grad)
                            client_grads[name] = scaled_grad
                        
                        elif attack_type == 'sign_flipping_attack':
                            # Flip the sign of the gradient
                            client_grads[name] = -base_grad
                        
                        elif attack_type == 'noise_attack':
                            # Add noise to the gradient
                            noise_factor = getattr(config, 'NOISE_FACTOR', 5.0)
                            noise = torch.randn_like(base_grad) * noise_factor
                            client_grads[name] = base_grad + noise
                        
                        elif attack_type == 'label_flipping':
                            # Simulate effect of label flipping (similar to sign flipping but less aggressive)
                            client_grads[name] = -base_grad * 0.5
                        
                        else:
                            # Default to normal gradient
                            client_grads[name] = base_grad
                        
                malicious_gradients.append(client_grads)
            
            # Combine honest and malicious gradients
            gradients = honest_gradients + malicious_gradients
            
            # Create feature vectors for clients
            # In real scenario these would be extracted from gradients
            # Here we create synthetic features
            features = torch.zeros((num_clients, 6), device=device)
            
            # Honest client features (good values for all features)
            for i in range(num_clients - num_malicious):
                # Features: [VAE recon error, root similarity, client similarity, norm, sign consistency, shapley]
                # Lower VAE recon error is better (closer to 0)
                features[i, 0] = torch.rand(1).item() * 0.3  # Low reconstruction error
                features[i, 1] = 0.7 + torch.rand(1).item() * 0.3  # High root similarity
                features[i, 2] = 0.7 + torch.rand(1).item() * 0.3  # High client similarity
                features[i, 3] = 0.4 + torch.rand(1).item() * 0.2  # Normal gradient norm
                features[i, 4] = 0.7 + torch.rand(1).item() * 0.3  # High sign consistency
                features[i, 5] = 0.7 + torch.rand(1).item() * 0.3  # High Shapley value
            
            # Malicious client features (based on attack type)
            for i in range(num_malicious):
                idx = num_clients - num_malicious + i
                
                # Base values (will be modified based on attack)
                features[idx, 0] = 0.6 + torch.rand(1).item() * 0.3  # High reconstruction error
                features[idx, 1] = 0.2 + torch.rand(1).item() * 0.3  # Low root similarity
                features[idx, 2] = 0.2 + torch.rand(1).item() * 0.3  # Low client similarity
                features[idx, 4] = 0.2 + torch.rand(1).item() * 0.3  # Low sign consistency
                features[idx, 5] = 0.1 + torch.rand(1).item() * 0.2  # Low Shapley value
                
                # Attack-specific feature modifications
                if attack_type == 'scaling_attack':
                    features[idx, 3] = 0.8 + torch.rand(1).item() * 0.2  # Very high gradient norm
                elif attack_type == 'partial_scaling_attack':
                    features[idx, 3] = 0.7 + torch.rand(1).item() * 0.2  # High gradient norm
                    features[idx, 0] = 0.5 + torch.rand(1).item() * 0.3  # Medium-high reconstruction error
                elif attack_type == 'sign_flipping_attack':
                    features[idx, 4] = 0.1 + torch.rand(1).item() * 0.2  # Very low sign consistency
                elif attack_type == 'noise_attack':
                    features[idx, 0] = 0.7 + torch.rand(1).item() * 0.2  # High reconstruction error
                    features[idx, 1] = 0.1 + torch.rand(1).item() * 0.2  # Very low root similarity
                elif attack_type == 'label_flipping':
                    features[idx, 4] = 0.3 + torch.rand(1).item() * 0.2  # Low sign consistency
                    features[idx, 3] = 0.4 + torch.rand(1).item() * 0.2  # Normal gradient norm
            
            # Save features for entropy calculation
            actor_critic.saved_features.append(features)
            
            # Get weights from actor network
            weights = actor_critic.select_action(features)
            
            # Simulate dual attention performance periodically to set baseline
            if episode % 5 == 0 or episode == 0:  # Reduced frequency from 20 to 5
                # Create dual attention weights (higher for honest clients, lower for malicious)
                dual_weights = torch.zeros(num_clients, device=device)
                for i in range(num_clients - num_malicious):
                    dual_weights[i] = 0.7 + torch.rand(1).item() * 0.3  # Higher weights for honest
                for i in range(num_malicious):
                    idx = num_clients - num_malicious + i
                    dual_weights[idx] = 0.1 + torch.rand(1).item() * 0.2  # Lower weights for malicious
                    
                # Normalize weights
                dual_weights = dual_weights / dual_weights.sum()
                
                # Evaluate dual attention performance
                env.restore_model_state()
                env.apply_gradients(gradients, dual_weights)
                try:
                    dual_loss, dual_acc = env.evaluate()
                    env.set_dual_attention_baseline(dual_loss, dual_acc)
                    
                    # Restore model state for RL agent
                    env.restore_model_state()
                except Exception as e:
                    print(f"Error evaluating dual attention: {str(e)}")
                    # Continue without setting baseline
                    continue
            
            # Apply gradients and get reward
            try:
                reward, done, info = env.step(gradients, weights, compare_to_baseline=True)
                
                # Store reward
                actor_critic.rewards.append(reward)
                
                # Update running reward
                running_reward = 0.05 * reward + 0.95 * running_reward
                
                # Anneal temperature
                if episode > episodes * 0.1:  # Start annealing after 10% of training
                    progress = min(1.0, (episode - episodes * 0.1) / (episodes * 0.9))
                    new_temp = initial_temp - progress * (initial_temp - min_temp)
                    actor_critic.actor.set_temperature(new_temp)
                
                # Print stats more selectively
                if episode % 2 == 0 or episode == episodes - 1:
                    print(f"Episode {episode+1}/{episodes} | Attack: {attack_type} | Reward: {reward:.4f}")
                    print(f"Temperature: {actor_critic.actor.temperature.item():.2f} | Val Loss: {info['val_loss']:.4f} | Val Acc: {info['val_acc']:.4f}")
                    if 'loss_vs_baseline' in info:
                        print(f"vs Baseline: Loss Diff = {info['loss_vs_baseline']:.4f}, Acc Diff = {info['acc_vs_baseline']:.4f}")
                
                # Track successful episodes
                successful_episodes += 1
                
            except Exception as e:
                print(f"Error in episode {episode+1}: {str(e)}")
                episode_errors += 1
                if episode_errors > 3:
                    print("Too many episode errors, stopping pre-training")
                    break
                continue
                
            # If accumulated enough experience, update policy
            if len(actor_critic.saved_log_probs) > 0 and (episode + 1) % 5 == 0:  # Reduced from 10 to 5
                try:
                    update_policy(actor_critic, actor_optimizer, critic_optimizer)
                except Exception as e:
                    print(f"Error updating policy: {str(e)}")
            
            # Save model periodically - but less frequently
            if (episode + 1) % max(episodes // 2, 1) == 0:
                save_dir = os.path.join('model_weights', 'rl_actor_critic')
                os.makedirs(save_dir, exist_ok=True)
                try:
                    torch.save(actor_critic.state_dict(), os.path.join(save_dir, f'actor_critic_episode_{episode+1}.pth'))
                except Exception as e:
                    print(f"Error saving model: {str(e)}")
        
        # Save final model
        save_dir = os.path.join('model_weights', 'rl_actor_critic')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(actor_critic.state_dict(), os.path.join(save_dir, 'actor_critic_final.pth'))
        print(f"Final RL model saved to {os.path.join(save_dir, 'actor_critic_final.pth')}")
        
    except Exception as e:
        print(f"Error during RL pre-training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"RL pre-training completed in {total_time:.2f} seconds with {successful_episodes} successful episodes")
    return actor_critic


def update_policy(actor_critic, actor_optimizer, critic_optimizer):
    """
    Update the policy using policy gradient methods with entropy bonus.
    
    Args:
        actor_critic: ActorCritic model
        actor_optimizer: Optimizer for actor network
        critic_optimizer: Optimizer for critic network
    """
    # Calculate returns
    returns = []
    R = 0
    gamma = config.RL_GAMMA if hasattr(config, 'RL_GAMMA') else 0.99
    
    # Calculate returns in reverse order
    for r in actor_critic.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    
    # Convert to tensor
    returns = torch.tensor(returns, device=actor_critic.actor.temperature.device)
    
    # Normalize returns for stability
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    
    # Actor loss with entropy bonus
    policy_loss = []
    entropy_term = 0
    
    # Get entropy coefficient from config
    entropy_coef = config.RL_ENTROPY_COEF if hasattr(config, 'RL_ENTROPY_COEF') else 0.01
    
    for i, (log_prob, R) in enumerate(zip(actor_critic.saved_log_probs, returns)):
        # Policy gradient loss
        policy_loss.append(-log_prob * R)
        
        # Add entropy bonus if available
        if hasattr(actor_critic.actor, 'get_entropy') and hasattr(actor_critic, 'saved_features'):
            # Calculate entropy using saved features
            if i < len(actor_critic.saved_features):
                features = actor_critic.saved_features[i]
                entropy = actor_critic.actor.get_entropy(features=features)
                entropy_term += entropy
    
    if policy_loss:
        # Combine policy loss with entropy bonus
        policy_loss = torch.stack(policy_loss).sum()
        
        # Add entropy bonus to encourage exploration
        if entropy_term != 0:
            policy_loss = policy_loss - entropy_coef * entropy_term
            print(f"Added entropy bonus: {entropy_coef * entropy_term:.4f}")
        
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
    if hasattr(actor_critic, 'saved_features'):
        actor_critic.saved_features = []