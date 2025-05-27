import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
from federated_learning.models.rl_actor_critic import ActorCritic
from federated_learning.training.rl_training import pretrain_rl_model
from federated_learning.config import config
from federated_learning.utils.data_utils import get_dataset


def main():
    """Train RL actor-critic model for gradient aggregation."""
    parser = argparse.ArgumentParser(description='Train RL model for gradient aggregation')
    parser.add_argument('--episodes', type=int, default=config.RL_PRETRAINING_EPISODES,
                        help='Number of episodes for pre-training')
    parser.add_argument('--input-dim', type=int, default=6,
                        help='Dimension of client feature vectors')
    parser.add_argument('--lr', type=float, default=config.RL_LEARNING_RATE,
                        help='Learning rate for training')
    parser.add_argument('--save-dir', type=str, default='model_weights/rl_actor_critic',
                        help='Directory to save model weights')
    parser.add_argument('--cpu', action='store_true', 
                        help='Use CPU instead of GPU')
    args = parser.parse_args()
    
    # Override config values with command line arguments
    config.RL_PRETRAINING_EPISODES = args.episodes
    config.RL_LEARNING_RATE = args.lr
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = config.device
    
    print(f"Using device: {device}")
    
    # Create directory to save model if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize actor-critic model
    actor_critic = ActorCritic(input_dim=args.input_dim)
    actor_critic = actor_critic.to(device)
    
    # Get dataset
    train_dataset, test_dataset = get_dataset(config.DATASET)
    
    # Create train and test loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Pretrain actor-critic model
    actor_critic = pretrain_rl_model(actor_critic, train_dataset, device=device)
    
    # Save final model
    torch.save(actor_critic.state_dict(), os.path.join(args.save_dir, 'actor_critic_final.pth'))
    print(f"Model saved to {os.path.join(args.save_dir, 'actor_critic_final.pth')}")


if __name__ == '__main__':
    main() 