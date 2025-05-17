import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import traceback
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.models.attention import DualAttention
from federated_learning.models.vae import GradientVAE

def create_test_gradient(size=1000, noise_level=0.0, malicious=False):
    """Create a test gradient with optional noise and malicious behavior"""
    base_grad = torch.randn(size)
    if malicious:
        # Create malicious gradient by either:
        # 1. Adding large noise
        # 2. Flipping signs
        # 3. Scaling up/down
        attack_type = np.random.choice(['noise', 'flip', 'scale'])
        if attack_type == 'noise':
            noise = torch.randn(size) * noise_level * 10  # Larger noise for malicious
            return base_grad + noise
        elif attack_type == 'flip':
            return -base_grad  # Sign flipping attack
        else:  # scale
            scale_factor = np.random.choice([0.1, 10.0])  # Either scale down or up
            return base_grad * scale_factor
    else:
        if noise_level > 0:
            noise = torch.randn(size) * noise_level
            return base_grad + noise
        return base_grad

def create_gradient_features(grad, vae, root_gradients):
    """Create feature vector for a gradient using VAE and other metrics"""
    # Normalize gradient
    norm = torch.norm(grad) + 1e-8
    normalized_grad = grad / norm
    
    # Get VAE reconstruction error
    with torch.no_grad():
        recon, _, _ = vae(normalized_grad.unsqueeze(0))
        re_val = F.mse_loss(recon, normalized_grad.unsqueeze(0), reduction='mean').item()
    
    # Calculate cosine similarity with root gradients
    cos_sims = [F.cosine_similarity(normalized_grad, r, dim=0).item() for r in root_gradients]
    mean_cos_sim = np.mean(cos_sims)
    
    # Calculate gradient norm
    grad_norm = torch.norm(grad).item()
    
    # Calculate gradient statistics
    grad_mean = grad.mean().item()
    grad_std = grad.std().item()
    
    # Combine features
    features = torch.tensor([
        re_val,           # VAE reconstruction error
        mean_cos_sim,     # Mean cosine similarity with root gradients
        0.0,             # Placeholder for neighbor similarity (will be updated during aggregation)
        grad_norm,        # Gradient norm
        grad_mean,        # Gradient mean
        grad_std          # Gradient standard deviation
    ])
    
    return features

def test_dual_attention_weighting():
    print("\n=== Testing Dual Attention Weighting ===")
    
    try:
        # Initialize VAE for feature extraction
        input_dim = 1000
        vae = GradientVAE(
            input_dim=input_dim,
            hidden_dim=256,
            latent_dim=128,
            dropout_rate=0.2,
            projection_dim=512
        )
        
        # Create root gradients (trusted gradients)
        print("\nGenerating root gradients...")
        root_gradients = [create_test_gradient(input_dim, noise_level=0.01) for _ in range(50)]
        root_gradients = [g / (torch.norm(g) + 1e-8) for g in root_gradients]  # Normalize
        
        # Train VAE on root gradients
        print("\nTraining VAE on root gradients...")
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        vae.train()
        
        for epoch in range(20):
            epoch_loss = 0
            for grad in root_gradients:
                optimizer.zero_grad()
                recon, mu, logvar = vae(grad.unsqueeze(0))
                loss = vae.loss_function(recon, grad.unsqueeze(0), mu, logvar)[0]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"VAE Epoch {epoch+1}/20, Loss: {epoch_loss/len(root_gradients):.4f}")
        
        # Create test gradients (mix of honest and malicious)
        print("\nGenerating test gradients...")
        num_clients = 20
        num_malicious = 4  # 20% malicious clients
        
        test_gradients = []
        is_malicious = []
        
        # Generate honest gradients
        for _ in range(num_clients - num_malicious):
            grad = create_test_gradient(input_dim, noise_level=0.1)
            test_gradients.append(grad)
            is_malicious.append(0)
        
        # Generate malicious gradients
        for _ in range(num_malicious):
            grad = create_test_gradient(input_dim, noise_level=0.1, malicious=True)
            test_gradients.append(grad)
            is_malicious.append(1)
        
        # Create feature vectors for all gradients
        print("\nCreating feature vectors...")
        feature_vectors = []
        for grad in test_gradients:
            features = create_gradient_features(grad, vae, root_gradients)
            feature_vectors.append(features)
        
        feature_vectors = torch.stack(feature_vectors)
        is_malicious = torch.tensor(is_malicious, dtype=torch.float32)
        
        # Initialize and train Dual Attention
        print("\nTraining Dual Attention...")
        dual_attention = DualAttention(
            feature_dim=6,  # Number of features
            hidden_dim=32,
            num_heads=4,
            dropout=0.2
        )
        
        # Create global context (average of root gradient features)
        global_context = torch.stack([
            create_gradient_features(g, vae, root_gradients) for g in root_gradients
        ]).mean(dim=0)
        
        # Train Dual Attention
        optimizer = torch.optim.Adam(dual_attention.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        num_epochs = 50
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            dual_attention.train()
            epoch_loss = 0
            predictions = []
            true_labels = []
            
            # Forward pass
            attention_weights = dual_attention(feature_vectors, global_context)
            
            # Calculate loss (binary cross entropy)
            loss = F.binary_cross_entropy(attention_weights, is_malicious)
            
            # Add separation loss to encourage clear distinction
            honest_weights = attention_weights[is_malicious == 0]
            malicious_weights = attention_weights[is_malicious == 1]
            if len(honest_weights) > 0 and len(malicious_weights) > 0:
                separation_loss = torch.mean(honest_weights) - torch.mean(malicious_weights)
                loss = loss + 0.1 * F.relu(0.5 - separation_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_attention.parameters(), 1.0)
            optimizer.step()
            
            # Update learning rate
            scheduler.step(loss)
            
            # Track predictions
            predictions.extend(attention_weights.detach().cpu().numpy())
            true_labels.extend(is_malicious.cpu().numpy())
            
            # Calculate metrics
            predictions = np.array(predictions)
            binary_preds = (predictions > 0.5).astype(int)
            true_labels = np.array(true_labels)
            
            accuracy = np.mean(binary_preds == true_labels)
            precision = precision_score(true_labels, binary_preds, zero_division=0)
            recall = recall_score(true_labels, binary_preds, zero_division=0)
            f1 = f1_score(true_labels, binary_preds, zero_division=0)
            
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}:")
                print(f"Loss: {loss.item():.4f}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = dual_attention.state_dict().copy()
        
        # Load best model
        if best_model_state is not None:
            dual_attention.load_state_dict(best_model_state)
        
        # Final evaluation
        print("\nFinal Evaluation:")
        dual_attention.eval()
        with torch.no_grad():
            final_weights = dual_attention(feature_vectors, global_context)
            final_preds = (final_weights > 0.5).float()
            
            # Calculate final metrics
            accuracy = (final_preds == is_malicious).float().mean().item()
            precision = precision_score(is_malicious.cpu(), final_preds.cpu(), zero_division=0)
            recall = recall_score(is_malicious.cpu(), final_preds.cpu(), zero_division=0)
            f1 = f1_score(is_malicious.cpu(), final_preds.cpu(), zero_division=0)
            
            print(f"Final Accuracy: {accuracy:.4f}")
            print(f"Final Precision: {precision:.4f}")
            print(f"Final Recall: {recall:.4f}")
            print(f"Final F1 Score: {f1:.4f}")
            
            # Analyze weight distribution
            honest_weights = final_weights[is_malicious == 0]
            malicious_weights = final_weights[is_malicious == 1]
            
            print("\nWeight Distribution:")
            print(f"Honest clients - Mean: {honest_weights.mean().item():.4f}, Std: {honest_weights.std().item():.4f}")
            print(f"Malicious clients - Mean: {malicious_weights.mean().item():.4f}, Std: {malicious_weights.std().item():.4f}")
            
            # Test gradient aggregation
            print("\nTesting Gradient Aggregation:")
            # Normalize weights for aggregation
            agg_weights = F.softmax(final_weights, dim=0)
            
            # Calculate weighted average
            weighted_grad = torch.zeros_like(test_gradients[0])
            for i, grad in enumerate(test_gradients):
                weighted_grad += grad * agg_weights[i]
            
            # Calculate cosine similarity with root average
            root_avg = torch.stack(root_gradients).mean(dim=0)
            cos_sim = F.cosine_similarity(weighted_grad, root_avg, dim=0).item()
            
            print(f"Cosine similarity with root average: {cos_sim:.4f}")
            
            # Calculate weight distribution in aggregation
            print("\nAggregation Weight Distribution:")
            print(f"Honest clients total weight: {agg_weights[is_malicious == 0].sum().item():.4f}")
            print(f"Malicious clients total weight: {agg_weights[is_malicious == 1].sum().item():.4f}")
            
            # Verify that malicious clients get lower weights
            assert agg_weights[is_malicious == 1].mean() < agg_weights[is_malicious == 0].mean(), \
                "Malicious clients should receive lower weights in aggregation"
            
            print("\nAll tests passed! Dual Attention successfully distinguishes between honest and malicious clients.")
            
    except Exception as e:
        print(f"\nError in test_dual_attention_weighting: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_dual_attention_weighting() 