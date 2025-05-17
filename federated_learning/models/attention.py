import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Set defaults for missing config parameters
try:
    from federated_learning.config.config import MALICIOUS_WEIGHTING_METHOD, SHAPLEY_WEIGHT, MALICIOUS_PENALTY_FACTOR
except ImportError:
    # Set defaults if not defined in config
    if 'MALICIOUS_WEIGHTING_METHOD' not in locals():
        MALICIOUS_WEIGHTING_METHOD = 'continuous'
        print(f"Warning: MALICIOUS_WEIGHTING_METHOD not defined in config, using default: {MALICIOUS_WEIGHTING_METHOD}")
    
    if 'SHAPLEY_WEIGHT' not in locals():
        SHAPLEY_WEIGHT = 0.5
        print(f"Warning: SHAPLEY_WEIGHT not defined in config, using default: {SHAPLEY_WEIGHT}")
        
    if 'MALICIOUS_PENALTY_FACTOR' not in locals():
        MALICIOUS_PENALTY_FACTOR = 0.5
        print(f"Warning: MALICIOUS_PENALTY_FACTOR not defined in config, using default: {MALICIOUS_PENALTY_FACTOR}")

class DualAttention(nn.Module):
    """
    Dual Attention mechanism for gradient evaluation.
    This model uses both self-attention and cross-attention to evaluate client gradients
    based on their feature vectors:
    1. Reconstruction error from VAE (normalized)
    2. Cosine similarity to root gradients (normalized)
    3. Cosine similarity to other clients (normalized)
    4. Gradient norm (normalized adaptively)
    5. Pattern consistency with root gradients (normalized)
    6. Shapley value (contribution to model performance) [optional]
    """
    
    def __init__(self, feature_dim=5, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initialize the Dual Attention model.
        
        Args:
            feature_dim: Dimension of input features (default: 5)
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(DualAttention, self).__init__()
        
        # Store dimensions
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Support for both 5 features (original) and 6 features (with Shapley values)
        self.supports_shapley = (feature_dim == 6)
        
        if feature_dim != 5 and feature_dim != 6:
            print(f"WARNING: Expected feature_dim=5 or feature_dim=6, got {feature_dim}. Adjusting to 5.")
            self.feature_dim = 5
            self.supports_shapley = False
        
        # Enhanced feature transformation layers with more expressive power
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),  # LeakyReLU instead of ReLU for better gradient flow
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Additional linear layer
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        
        # Self-attention layer (ensuring batch_first is set for PyTorch 1.9+)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Additional transformer layers if num_layers > 1
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_layers - 1):
            self.transformer_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
            )
        
        # Enhanced feature interaction layer
        self.feature_interaction = PairwiseFeatureInteraction(hidden_dim)
        
        # Specific attention for malicious feature patterns
        self.malicious_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1)
        )
        
        # Output layers with enhanced expressivity
        self.trust_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimator with enhanced layers
        self.confidence_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Malicious pattern detection module with more expressive layers
        self.malicious_pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Direct feature interpretation layer
        self.feature_importance = nn.Parameter(torch.ones(feature_dim) / feature_dim)
        
        # Initialize weights for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, global_context=None):
        """
        Compute trust scores and confidence scores for gradient features.
        
        Args:
            features: Client gradient features [batch_size, feature_dim]
            global_context: Optional global context tensor [1, feature_dim]
            
        Returns:
            malicious_scores: Malicious scores [batch_size, 1]
            confidence_scores: Confidence scores [batch_size, 1]
        """
        batch_size = features.size(0)
        device = features.device
        
        # Initialize confidence_scores to default value of ones
        confidence_scores = torch.ones(batch_size, 1, device=device)
        
        # Check feature dimension
        if features.size(1) != self.feature_dim:
            print(f"WARNING: Input feature dimension ({features.size(1)}) does not match expected dimension ({self.feature_dim})")
            # Dynamically adjust if possible, otherwise pad or truncate
            if features.size(1) < self.feature_dim:
                # Pad with zeros
                padding = torch.zeros(batch_size, self.feature_dim - features.size(1), device=device)
                features = torch.cat([features, padding], dim=1)
                print(f"Padded features to shape: {features.shape}")
            elif features.size(1) > self.feature_dim:
                # Truncate
                features = features[:, :self.feature_dim]
                print(f"Truncated features to shape: {features.shape}")
        
        # Apply feature importance weighting directly
        # This gives more weight to more important features (like norm and Shapley values)
        # Specifically increase weight of gradient norm (index 3) and Shapley (index 5 if present)
        weighted_features = features.clone()
        
        # IMPORTANT: Enhance feature contrast to prevent constant outputs
        # Emphasize important features for malicious detection
        
        # Feature 0: Reconstruction error - higher is more suspicious
        if features.size(1) > 0:
            weighted_features[:, 0] = torch.sigmoid((features[:, 0] - 0.4) * 4)
        
        # Feature 1: Root similarity - lower is more suspicious 
        if features.size(1) > 1:
            weighted_features[:, 1] = 1.0 - torch.sigmoid((features[:, 1] - 0.6) * 4)
            
        # Feature 2: Client similarity - lower is more suspicious
        if features.size(1) > 2:
            weighted_features[:, 2] = 1.0 - torch.sigmoid((features[:, 2] - 0.6) * 4)
            
        # Feature 3: Gradient norm - higher is more suspicious
        if features.size(1) > 3:
            weighted_features[:, 3] = torch.sigmoid((features[:, 3] - 0.6) * 4)
            
        # Feature 4: Consistency - lower is more suspicious
        if features.size(1) > 4:
            weighted_features[:, 4] = 1.0 - torch.sigmoid((features[:, 4] - 0.6) * 4)
            
        # Feature 5: Shapley - if present, higher is more suspicious
        if features.size(1) > 5 and self.supports_shapley:
            weighted_features[:, 5] = torch.sigmoid((features[:, 5] - 0.5) * 4)
        
        # Direct feature interpretation for initial malicious score estimate
        # Higher values indicate more likely to be malicious
        direct_malicious_score = torch.sigmoid(torch.sum(weighted_features * self.feature_importance, dim=1))
        
        # If no global context provided, use mean of features
        if global_context is None:
            global_context = torch.mean(features, dim=0, keepdim=True)
            
        # Project features to hidden dimension
        x = self.feature_projection(features)
        
        # Project global context to hidden dimension
        context = self.feature_projection(global_context)
        
        # Self-attention
        self_attn_output, self_attn_weights = self.self_attention(x, x, x)
        
        # Cross-attention with global context
        cross_attn_output, cross_attn_weights = self.cross_attention(x, context, context)
        
        # Combine self-attention and cross-attention (residual connection)
        combined = x + self_attn_output + cross_attn_output
        
        # Apply additional transformer layers if available
        for layer in self.transformer_layers:
            combined = layer(combined)
        
        # Apply pairwise feature interaction
        interaction_output = self.feature_interaction(combined)
        
        # Compute attention to malicious patterns
        malicious_attention = self.malicious_attention(combined)
        
        # Add residual connection
        final_features = combined + interaction_output
        
        # Compute malicious scores - now higher means more likely malicious
        malicious_scores = self.trust_output(final_features)
        
        # Compute confidence scores
        confidence_scores = self.confidence_output(final_features)
        
        # Squeeze to remove singleton dimension
        malicious_scores = malicious_scores.squeeze(-1)
        confidence_scores = confidence_scores.squeeze(-1)
        
        # Apply direct feature importance as a calibration factor
        calibrated_malicious = (malicious_scores + direct_malicious_score) / 2
        
        # Extra adjustment: Enhance high gradient norm penalty
        if features.size(1) > 3:  # Ensure we have gradient norm feature
            # Increase malicious score for clients with high gradient norms
            high_norm_boost = torch.sigmoid((features[:, 3] - 0.6) * 8) * 0.2
            calibrated_malicious = torch.clamp(calibrated_malicious + high_norm_boost, 0, 1)
        
        return calibrated_malicious, confidence_scores

    def get_gradient_weights(self, features, malicious_scores=None, confidence_scores=None):
        """
        Compute weights for gradient aggregation based on malicious scores and features.
        
        Args:
            features: Gradient features tensor [num_clients, num_features]
            malicious_scores: Pre-computed malicious scores (optional)
            confidence_scores: Pre-computed confidence scores (optional)
            
        Returns:
            weights: Aggregation weights [num_clients]
            malicious_indices: Indices of detected malicious clients
        """
        device = features.device
        num_clients = features.shape[0]
        
        # If malicious scores not provided, compute them
        if malicious_scores is None:
            malicious_scores, confidence_scores = self(features)
        
        # If confidence scores still None, use ones
        if confidence_scores is None:
            confidence_scores = torch.ones_like(malicious_scores)
        
        # Get values as numpy for outlier detection (more robust with small batches)
        malicious_values = malicious_scores.detach().cpu().numpy()
        
        # Use continuous weighting based on inverse of malicious scores
        if MALICIOUS_WEIGHTING_METHOD == 'continuous':
            # Calculate mean and std of malicious scores
            mean_malicious = malicious_scores.mean()
            std_malicious = malicious_scores.std()
            
            # Initialize weights as inverse of malicious scores 
            # Higher malicious score = lower weight (1 - score)
            weights = 1.0 - malicious_scores.clone()
            
            # Find potential malicious clients based on malicious scores
            threshold = mean_malicious + std_malicious / 2
            malicious_indices = [i for i, score in enumerate(malicious_values) if score > threshold]
            
            # Check for abnormally high gradient norms (Feature index 3)
            norm_features = features[:, 3].detach().cpu().numpy()  # Feature index 3 is gradient norm
            norm_mean = np.mean(norm_features)
            norm_std = np.std(norm_features)
            
            # More sensitive threshold for high norm detection
            norm_threshold = norm_mean + 0.7 * norm_std  # Reduced from 1.0 * std to 0.7 * std
            
            # Find clients with abnormally high gradient norms
            high_norm_indices = []
            for i, norm in enumerate(norm_features):
                if norm > norm_threshold:
                    high_norm_indices.append(i)
                    print(f"Detected client {i} as suspicious due to high gradient norm: {norm:.4f} (threshold: {norm_threshold:.4f})")
                    print(f"  This is {(norm - norm_mean) / norm_std:.2f} standard deviations above the mean")
            
            # Combine malicious indices from both methods
            malicious_indices = list(set(malicious_indices + high_norm_indices))
            
            # Sort for consistent output
            malicious_indices.sort()
            
            # Apply penalty to suspicious clients based on MALICIOUS_PENALTY_FACTOR
            # Higher penalty factor means greater penalty for suspicious clients
            if malicious_indices and MALICIOUS_PENALTY_FACTOR > 0:
                for idx in malicious_indices:
                    # If it's a high norm client, apply stronger penalty
                    if idx in high_norm_indices:
                        # Calculate how much this client's norm exceeds the threshold
                        severity = (norm_features[idx] - norm_threshold) / (norm_threshold + 1e-5)
                        # Cap severity at a reasonable level
                        severity = min(severity, 5.0)
                        
                        # Scale penalty strength based on severity and penalty factor
                        penalty_strength = min(0.98, MALICIOUS_PENALTY_FACTOR * (1.0 + severity * 0.5))
                        
                        # Apply penalty (reduce weight by penalty_strength percentage)
                        weights[idx] = weights[idx] * (1 - penalty_strength)
                        
                        print(f"Applied strong penalty of {penalty_strength:.4f} to client {idx} due to high gradient norm")
                        print(f"  Weight reduced from {(1.0 - malicious_scores[idx]).item():.4f} to {weights[idx].item():.4f}")
                    else:
                        # For detection based on malicious score, apply standard penalty
                        distance_from_mean = (malicious_scores[idx] - mean_malicious) / (std_malicious + 1e-5)
                        
                        # Cap distance at a reasonable value
                        distance_from_mean = min(distance_from_mean, 3.0)
                        
                        # Scale penalty based on distance and penalty factor
                        penalty = 1.0 - torch.exp(-distance_from_mean * MALICIOUS_PENALTY_FACTOR)
                        penalty = min(penalty, torch.tensor(0.9).to(device))
                        
                        # Apply penalty
                        old_weight = weights[idx].item()
                        weights[idx] = weights[idx] * (1 - penalty)
                        
                        print(f"Applied penalty of {penalty.item():.4f} to client {idx} based on malicious score")
                        print(f"  Weight reduced from {old_weight:.4f} to {weights[idx].item():.4f}")
                
                print(f"Applied penalties with factor {MALICIOUS_PENALTY_FACTOR} to {len(malicious_indices)} suspicious clients")
            
            # Ensure minimum weight (never completely exclude a client)
            min_weight = 0.01
            weights = torch.clamp(weights, min=min_weight)
            
            # Apply softmax to normalize weights to sum to 1
            # Use temperature parameter to control sharpness
            temperature = 1.0
            weights = F.softmax(weights / temperature, dim=0)
            
        else:
            # Use threshold-based weighting
            threshold = 0.5
            malicious_mask = malicious_scores > threshold
            malicious_indices = torch.nonzero(malicious_mask).flatten().tolist()
            
            # Create weights
            weights = torch.zeros_like(malicious_scores)
            weights[~malicious_mask] = 1.0 - malicious_scores[~malicious_mask]  # Honest clients get higher weights
            
            # Apply strong penalty to malicious clients based on MALICIOUS_PENALTY_FACTOR
            # Higher penalty factor means lower weights for malicious clients
            malicious_weight = 0.01 * (1 - MALICIOUS_PENALTY_FACTOR)  # Minimum weight decreases as penalty increases
            weights[malicious_mask] = malicious_weight
            
            # Normalize weights to sum to 1
            weights = weights / weights.sum() if weights.sum() > 0 else torch.ones_like(weights) / num_clients
            
        # Print detection summary
        print(f"Detected {len(malicious_indices)} potential malicious clients")
        if malicious_indices:
            print(f"Malicious client indices: {malicious_indices}")
        
        # Print weight statistics
        print(f"Min weight: {weights.min().item():.4f}")
        print(f"Max weight: {weights.max().item():.4f}")
        print(f"Mean weight: {weights.mean().item():.4f}")
        print(f"Std weight: {weights.std().item():.4f}")
        
        # Print malicious score distribution
        print("\nMalicious Score Distribution:")
        print(f"Mean: {malicious_scores.mean().item():.4f}")
        print(f"Std: {malicious_scores.std().item():.4f}")
        print(f"Min: {malicious_scores.min().item():.4f}")
        print(f"Max: {malicious_scores.max().item():.4f}")
        
        # Print confidence score distribution if available
        if confidence_scores is not None:
            print("\nConfidence Score Distribution:")
            print(f"Mean: {confidence_scores.mean().item():.4f}")
            print(f"Std: {confidence_scores.std().item():.4f}")
            print(f"Min: {confidence_scores.min().item():.4f}")
            print(f"Max: {confidence_scores.max().item():.4f}")
        
        return weights, malicious_indices

class PairwiseFeatureInteraction(nn.Module):
    """
    Learns pairwise interactions between feature dimensions.
    This helps capture complex relationships between different gradient features.
    """
    def __init__(self, hidden_dim):
        super(PairwiseFeatureInteraction, self).__init__()
        
        # Interaction matrix to learn relationships between features
        self.interaction_matrix = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        
        # Feature gating to control information flow
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation after interaction
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Apply pairwise feature interaction.
        
        Args:
            x: Input features [batch_size, hidden_dim]
            
        Returns:
            output: Transformed features with interaction [batch_size, hidden_dim]
        """
        # Apply feature gating
        gate = self.feature_gate(x)
        
        # Apply pairwise interactions
        interaction = torch.matmul(x, self.interaction_matrix)
        
        # Gate the interactions
        gated_interaction = interaction * gate
        
        # Residual connection
        combined = x + gated_interaction
        
        # Transform
        output = self.transform(combined)
        
        return output 