import torch
import torch.nn as nn
import numpy as np
from federated_learning.training.training_utils import client_update
from federated_learning.models.attention import DualAttention
from federated_learning.training.aggregation import aggregate_gradients, extract_gradient_features
from federated_learning.config.config import *

class SimpleTestModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=2):
        super(SimpleTestModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def generate_test_scenario():
    """Generate a test scenario with known patterns"""
    # Create root model and gradient
    model = SimpleTestModel()
    
    # Generate root gradient with consistent direction
    root_data = torch.randn(32, 10)
    root_labels = torch.randint(0, 2, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Generate root gradient
    output = model(root_data)
    loss = criterion(output, root_labels)
    loss.backward()
    root_gradient = {name: param.grad.clone() for name, param in model.named_parameters()}
    
    # Calculate root gradient norm
    root_grad_tensor = torch.cat([p.flatten() for p in root_gradient.values()])
    root_norm = torch.norm(root_grad_tensor)
    
    # Reset model gradients
    model.zero_grad()
    
    # Generate client updates
    client_gradients = []
    client_types = []  # 0: honest, 1: scale attack, 2: sign flip
    
    # Honest clients (3)
    for _ in range(3):
        # Create similar but not identical data
        client_data = root_data + torch.randn_like(root_data) * 0.1
        client_labels = root_labels.clone()
        
        # Generate gradient
        output = model(client_data)
        loss = criterion(output, client_labels)
        loss.backward()
        
        # Add small random noise to gradient
        grad = {name: param.grad.clone() + torch.randn_like(param.grad) * 0.1 * root_norm 
               for name, param in model.named_parameters()}
        
        client_gradients.append(grad)
        client_types.append(0)
        model.zero_grad()
    
    # Scale attack (2)
    for scale in [5.0, 10.0]:  # Two different scales
        output = model(root_data)
        loss = criterion(output, root_labels)
        loss.backward()
        
        # Scale the gradient
        scale_grad = {name: param.grad.clone() * scale
                     for name, param in model.named_parameters()}
        client_gradients.append(scale_grad)
        client_types.append(1)
        model.zero_grad()
    
    # Sign flip attack (2)
    for scale in [1.0, 1.5]:  # Two different magnitudes
        output = model(root_data)
        loss = criterion(output, root_labels)
        loss.backward()
        
        # Flip signs and scale
        flip_grad = {name: -param.grad.clone() * scale
                    for name, param in model.named_parameters()}
        client_gradients.append(flip_grad)
        client_types.append(2)
        model.zero_grad()
    
    return model, root_gradient, client_gradients, client_types

def train_dual_attention(features, labels, epochs=100):
    """Train the DualAttention model on the given features."""
    model = DualAttention(
        feature_dim=features.shape[1],
        hidden_dim=DUAL_ATTENTION_HIDDEN_DIM * 2,  # Larger model
        num_heads=4,
        dropout=DUAL_ATTENTION_DROPOUT
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.BCELoss()
    
    # Convert labels to float
    labels = labels.float()
    
    # Calculate class weights to handle imbalance
    num_honest = (labels == 0).sum()
    num_malicious = (labels == 1).sum()
    total = len(labels)
    
    honest_weight = total / (2 * num_honest) if num_honest > 0 else 1.0
    malicious_weight = total / (2 * num_malicious) if num_malicious > 0 else 1.0
    
    sample_weights = torch.where(labels == 0, 
                               torch.tensor(honest_weight).clone().detach(), 
                               torch.tensor(malicious_weight).clone().detach())
    
    # Training loop
    best_loss = float('inf')
    patience = 20  # Longer patience
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        trust_scores, confidence = model(features)
        
        # Calculate weighted loss
        bce_loss = criterion(trust_scores, labels)
        weighted_loss = (bce_loss * sample_weights).mean()
        
        # Add confidence regularization
        confidence_target = (trust_scores.detach() > 0.5).float()
        confidence_loss = criterion(confidence, confidence_target)
        
        # Add separation loss to encourage clear separation
        honest_scores = trust_scores[labels == 0]
        malicious_scores = trust_scores[labels == 1]
        
        if len(honest_scores) > 0 and len(malicious_scores) > 0:
            honest_mean = honest_scores.mean()
            malicious_mean = malicious_scores.mean()
            separation_loss = torch.relu(0.7 - (malicious_mean - honest_mean))  # Larger margin
            
            # Add variance loss to encourage tight clusters
            honest_var = honest_scores.var()
            malicious_var = malicious_scores.var()
            variance_loss = (honest_var + malicious_var) * 0.1
        else:
            separation_loss = torch.tensor(0.0)
            variance_loss = torch.tensor(0.0)
        
        # Total loss with adjusted weights
        total_loss = (weighted_loss * 1.0 + 
                     confidence_loss * 0.2 + 
                     separation_loss * 0.5 + 
                     variance_loss * 0.1)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate
        scheduler.step(total_loss)
        
        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")
            print(f"  BCE Loss: {weighted_loss.item():.4f}")
            print(f"  Confidence Loss: {confidence_loss.item():.4f}")
            print(f"  Separation Loss: {separation_loss.item():.4f}")
            print(f"  Variance Loss: {variance_loss.item():.4f}")
            
            # Print score distributions
            if len(honest_scores) > 0:
                print(f"  Honest scores - Mean: {honest_scores.mean():.4f}, Std: {honest_scores.std():.4f}")
            if len(malicious_scores) > 0:
                print(f"  Malicious scores - Mean: {malicious_scores.mean():.4f}, Std: {malicious_scores.std():.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def run_evaluation_tests():
    print("=== Running Evaluation Tests ===")
    
    try:
        # Generate test scenario once
        model, root_grad, client_grads, client_types = generate_test_scenario()
        
        # Convert gradients to flat tensors
        flat_root = torch.cat([p.flatten() for p in root_grad.values()])
        flat_clients = [torch.cat([p.flatten() for p in grad.values()]) 
                       for grad in client_grads]
        
        # Test 1: Feature Extraction
        print("\n=== Testing Gradient Feature Extraction ===")
        features = extract_gradient_features(flat_clients, flat_root)
        
        print("\nFeature Analysis:")
        print(f"Feature shape: {features.shape}")
        
        # Analyze features for each client type
        for i, (feat, client_type) in enumerate(zip(features, client_types)):
            client_desc = ["Honest", "Scale Attack", "Sign Flip"][client_type]
            cosine_sim = feat[0].item()
            norm_ratio = feat[1].item()
            relative_norm = feat[2].item()
            l2_dist = feat[3].item()
            
            print(f"\nClient {i} ({client_desc}):")
            print(f"Cosine Similarity: {cosine_sim:.4f}")
            print(f"Norm Ratio: {norm_ratio:.4f}")
            print(f"Relative Norm: {relative_norm:.4f}")
            print(f"L2 Distance: {l2_dist:.4f}")
            
            # Verify expected patterns
            if client_type == 0:  # Honest
                assert cosine_sim > 0.7, "Honest client should have positive cosine similarity"
                assert norm_ratio < 0.3, "Honest client should have relatively small norm"
                assert 0.5 < relative_norm < 2.0, "Honest client should have reasonable relative norm"
            elif client_type == 1:  # Scale attack
                assert norm_ratio > 0.8, "Scale attack should have large norm ratio"
                assert relative_norm > 5.0, "Scale attack should have large relative norm"
                assert cosine_sim > 0.9, "Scale attack should maintain direction"
            elif client_type == 2:  # Sign flip
                assert cosine_sim < -0.7, "Sign flip should have negative cosine similarity"
                assert l2_dist > 1.4, "Sign flip should be far in normalized space"
                assert relative_norm > 1.0, "Sign flip should have increased magnitude"
        
        # Test 2: Weight Assignment
        print("\n=== Testing Weight Assignment ===")
        
        # Create labels for training (0 for honest, 1 for malicious)
        labels = torch.tensor([1 if t > 0 else 0 for t in client_types])
        
        # Train the DualAttention model
        print("\nTraining DualAttention model...")
        dual_attention = train_dual_attention(features, labels)
        
        # Get weights
        weights = dual_attention.get_gradient_weights(features)
        
        print("\nWeight Analysis:")
        total_honest_weight = 0.0
        num_honest = 0
        
        for i, (weight, client_type) in enumerate(zip(weights, client_types)):
            client_desc = ["Honest", "Scale Attack", "Sign Flip"][client_type]
            print(f"Client {i} ({client_desc}): Weight = {weight.item():.4f}")
            
            if client_type == 0:
                total_honest_weight += weight.item()
                num_honest += 1
        
        # Verify weight patterns
        avg_honest_weight = total_honest_weight / num_honest if num_honest > 0 else 0
        print(f"\nAverage honest client weight: {avg_honest_weight:.4f}")
        
        assert avg_honest_weight > 0.2, "Honest clients should have significant average weight"
        assert abs(weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
        
        # Test 3: Aggregation
        print("\n=== Testing Aggregation ===")
        
        # Test different aggregation methods
        methods = ['fedavg', 'direct']
        results = {}
        
        for method in methods:
            print(f"\nTesting {method} aggregation:")
            
            if method == 'direct':
                # Aggregate with attention
                aggregated = aggregate_gradients(
                    flat_clients,
                    aggregation_method='direct',
                    root_gradient=flat_root,
                    dual_attention=dual_attention
                )
            else:
                # Simple averaging
                aggregated = aggregate_gradients(flat_clients, aggregation_method='fedavg')
            
            # Analyze result
            cosine_sim = torch.nn.functional.cosine_similarity(
                aggregated.unsqueeze(0),
                flat_root.unsqueeze(0)
            ).item()
            grad_norm = torch.norm(aggregated).item()
            
            print(f"Cosine similarity with root: {cosine_sim:.4f}")
            print(f"Gradient norm: {grad_norm:.4f}")
            
            results[method] = {
                'gradient': aggregated,
                'cosine_sim': cosine_sim,
                'norm': grad_norm
            }
        
        # Compare methods
        print("\nComparison:")
        fedavg_sim = results['fedavg']['cosine_sim']
        direct_sim = results['direct']['cosine_sim']
        print(f"FedAvg vs Root similarity: {fedavg_sim:.4f}")
        print(f"Direct vs Root similarity: {direct_sim:.4f}")
        
        # Direct method should be more similar to root
        assert direct_sim > fedavg_sim, "Direct method should better preserve root gradient direction"
        
        print("\n✓ All evaluation tests passed successfully!")
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    run_evaluation_tests() 