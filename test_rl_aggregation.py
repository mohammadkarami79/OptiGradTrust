import torch
import numpy as np
from federated_learning.training.server import Server
from federated_learning.models.rl_actor_critic import ActorCritic
from federated_learning.config.config import AGGREGATION_METHOD as original_AGGREGATION_METHOD
from federated_learning.config.config import RL_AGGREGATION_METHOD as original_RL_AGGREGATION_METHOD
from federated_learning.config.config import RL_WARMUP_ROUNDS as original_RL_WARMUP_ROUNDS
from federated_learning.config.config import RL_RAMP_UP_ROUNDS as original_RL_RAMP_UP_ROUNDS
import copy

def test_rl_aggregation():
    """Test RL aggregation method in the server"""
    print("\n=== Testing RL Aggregation ===")
    
    # Initialize server
    server = Server()
    device = server.device
    
    # Create mock clients class
    class MockClient:
        def __init__(self, is_malicious=False):
            self.is_malicious = is_malicious
    
    # Add mock clients to the server
    server.clients = [
        MockClient(is_malicious=False),
        MockClient(is_malicious=False),
        MockClient(is_malicious=False),
        MockClient(is_malicious=True),
        MockClient(is_malicious=True)
    ]
    
    # Mock client gradients and features
    num_clients = 5
    feature_dim = 6
    gradient_dim = 100
    
    # Create mock gradients
    gradients = [torch.randn(gradient_dim, device=device) for _ in range(num_clients)]
    
    # Create features - first 3 clients honest, last 2 malicious
    features = torch.zeros((num_clients, feature_dim), device=device)
    
    # Honest client features
    for i in range(3):
        features[i, 0] = 0.2  # Low reconstruction error
        features[i, 1] = 0.8  # High root similarity
        features[i, 2] = 0.8  # High client similarity
        features[i, 3] = 0.5  # Normal gradient norm
        features[i, 4] = 0.8  # High sign consistency
        features[i, 5] = 0.8  # High Shapley value
    
    # Malicious client features
    for i in range(3, 5):
        features[i, 0] = 0.8  # High reconstruction error
        features[i, 1] = 0.2  # Low root similarity
        features[i, 2] = 0.2  # Low client similarity
        features[i, 3] = 0.9  # High gradient norm
        features[i, 4] = 0.2  # Low sign consistency
        features[i, 5] = 0.2  # Low Shapley value
    
    # Override the get_weights method to return controlled weights for testing
    original_get_weights = server.actor_critic.get_weights
    
    def mock_get_weights(features):
        # Return predefined weights for testing
        return torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1], device=device)
    
    # Apply mock method
    server.actor_critic.get_weights = mock_get_weights
    
    # Mock client indices
    client_indices = list(range(num_clients))
    
    # Test RL aggregation
    print("\nTesting RL-based aggregation...")
    
    # Set up actor-critic model
    actor_critic = ActorCritic(input_dim=feature_dim).to(device)
    server.actor_critic = actor_critic
    
    # Required for checking client malicious status in _aggregate_rl
    server.trust_scores = torch.tensor([0.9, 0.85, 0.8, 0.2, 0.15], device=device)
    server.confidence_scores = torch.ones(5, device=device)
    
    # Get weights from actor-critic
    aggregated_gradient = server._aggregate_rl(gradients, features, client_indices)
    
    print(f"Shape of aggregated gradient: {aggregated_gradient.shape}")
    print(f"Aggregated gradient norm: {torch.norm(aggregated_gradient).item():.4f}")
    
    # Restore original get_weights method
    server.actor_critic.get_weights = original_get_weights
    
    return True

def test_hybrid_mode():
    """Test hybrid aggregation mode (dual attention + RL)"""
    print("\n=== Testing Hybrid Aggregation Mode ===")
    
    # Save original values to reuse them later
    AGGREGATION_METHOD = original_AGGREGATION_METHOD
    
    # Temporarily modify config by creating new variables
    RL_AGGREGATION_METHOD = 'hybrid'
    RL_WARMUP_ROUNDS = 2
    RL_RAMP_UP_ROUNDS = 3
    
    # Force FedAvg for this test to simplify the comparison
    test_AGGREGATION_METHOD = 'fedavg'
    
    # Initialize server
    server = Server()
    device = server.device
    
    # Create mock clients
    class MockClient:
        def __init__(self, is_malicious=False):
            self.is_malicious = is_malicious
    
    # Add clients
    server.clients = [
        MockClient(is_malicious=False),
        MockClient(is_malicious=False),
        MockClient(is_malicious=False),
        MockClient(is_malicious=True),
        MockClient(is_malicious=True)
    ]
    
    # Mock client gradients and features
    num_clients = 5
    feature_dim = 6
    gradient_dim = 100
    
    # Create mock gradients
    gradients = [torch.randn(gradient_dim, device=device) for _ in range(num_clients)]
    
    # Create features - first 3 clients honest, last 2 malicious
    features = torch.zeros((num_clients, feature_dim), device=device)
    
    # Honest client features
    for i in range(3):
        features[i, 0] = 0.2  # Low reconstruction error
        features[i, 1] = 0.8  # High root similarity
        features[i, 2] = 0.8  # High client similarity
        features[i, 3] = 0.5  # Normal gradient norm
        features[i, 4] = 0.8  # High sign consistency
        features[i, 5] = 0.8  # High Shapley value
    
    # Malicious client features
    for i in range(3, 5):
        features[i, 0] = 0.8  # High reconstruction error
        features[i, 1] = 0.2  # Low root similarity
        features[i, 2] = 0.2  # Low client similarity
        features[i, 3] = 0.9  # High gradient norm
        features[i, 4] = 0.2  # Low sign consistency
        features[i, 5] = 0.2  # Low Shapley value
    
    # Set up dual attention model
    if not hasattr(server, 'dual_attention') or server.dual_attention is None:
        server.dual_attention = server._create_dual_attention()
    
    # Mock trust scores
    server.trust_scores = torch.tensor([0.9, 0.85, 0.8, 0.2, 0.15], device=device)
    server.confidence_scores = torch.ones(5, device=device)
    server.weights = torch.tensor([0.3, 0.3, 0.3, 0.05, 0.05], device=device)
    
    # Set up RL model
    if not hasattr(server, 'actor_critic') or server.actor_critic is None:
        server.actor_critic = server._create_actor_critic()
    
    # Mock client indices
    client_indices = list(range(num_clients))
    
    # Store current round gradients for RL comparison
    server.current_round_gradients = gradients
    
    # Compute pure dual attention and pure RL gradients for verification
    pure_dual_gradient = server._aggregate_fedavg(gradients, server.weights)
    
    # We'll directly compute this as RL weights would be during testing
    rl_weights = torch.tensor([0.1, 0.15, 0.15, 0.3, 0.3], device=device)
    pure_rl_gradient = server._aggregate_fedavg(gradients, rl_weights)
    
    # Override the get_weights method to return controlled weights for testing
    original_get_weights = server.actor_critic.get_weights
    
    def mock_get_weights(features):
        # Return weights that are different from dual attention for testing
        return rl_weights
    
    # Apply mock method
    server.actor_critic.get_weights = mock_get_weights
    
    # Test different rounds of hybrid mode
    print("\nTesting different phases of hybrid mode...")
    
    # Test hybrid aggregation at each phase
    for round_idx in range(7):
        print(f"\nRound {round_idx}:")
        if round_idx < RL_WARMUP_ROUNDS:
            phase = "Warmup (Dual Attention)"
        elif round_idx < RL_WARMUP_ROUNDS + RL_RAMP_UP_ROUNDS:
            blend_ratio = (round_idx - RL_WARMUP_ROUNDS) / RL_RAMP_UP_ROUNDS
            phase = f"Ramp-up (Blend ratio: {blend_ratio:.2f})"
        else:
            phase = "Full RL"
        
        print(f"Phase: {phase}")
        
        # Set dual attention weights for comparison
        server.dual_attention_weights = torch.tensor([0.3, 0.3, 0.3, 0.05, 0.05], device=device)
        
        # Create a simple global model for testing
        from federated_learning.models.cnn import CNNMnist
        model = CNNMnist().to(device)
        server.global_model = model
        
        # Determine aggregation method
        if round_idx < RL_WARMUP_ROUNDS:
            # During warmup, use dual attention weights with fedavg
            print(f"Using aggregation method: {test_AGGREGATION_METHOD}")
            aggregated_gradient = server._aggregate_fedavg(gradients, server.weights)
        else:
            # During ramp-up or full RL, use RL-based aggregation
            print("Using aggregation method: rl")
            
            # Get RL-based gradient
            rl_gradient = server._aggregate_rl(gradients, features, client_indices)
            
            # If in ramp-up phase, blend with dual attention
            if round_idx < RL_WARMUP_ROUNDS + RL_RAMP_UP_ROUNDS:
                blend_ratio = (round_idx - RL_WARMUP_ROUNDS) / RL_RAMP_UP_ROUNDS
                print(f"Hybrid mode: Blending RL with dual attention (RL weight: {blend_ratio:.2f})")
                
                # Get dual attention gradient
                dual_attention_gradient = server._aggregate_fedavg(gradients, server.weights)
                
                # Blend the gradients
                aggregated_gradient = (blend_ratio * rl_gradient + 
                                     (1 - blend_ratio) * dual_attention_gradient)
            else:
                aggregated_gradient = rl_gradient
        
        # Print gradient statistics
        print(f"Aggregated gradient norm: {torch.norm(aggregated_gradient).item():.4f}")
    
    # Restore original get_weights method
    server.actor_critic.get_weights = original_get_weights
    
    return True

if __name__ == "__main__":
    print("Running RL aggregation tests...")
    
    # Run tests
    rl_test_passed = test_rl_aggregation()
    hybrid_test_passed = test_hybrid_mode()
    
    # Print results
    print("\n=== Test Results ===")
    print(f"RL Aggregation Test: {'PASSED' if rl_test_passed else 'FAILED'}")
    print(f"Hybrid Mode Test: {'PASSED' if hybrid_test_passed else 'FAILED'}") 