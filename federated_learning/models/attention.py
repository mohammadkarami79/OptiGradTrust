import torch
import torch.nn as nn

class DualAttention(nn.Module):
    def __init__(self, feature_size):
        super(DualAttention, self).__init__()
        # Enhanced architecture with deeper layers for better representation
        self.self_attention = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_size)
        )
        
        # More sophisticated cross-attention mechanism
        self.cross_attention = nn.Sequential(
            nn.Linear(feature_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Additional binary classifier to detect malicious vs benign
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X, X_ref):
        # Self-attention to capture internal patterns
        self_attended = self.self_attention(X)
        
        # Cross-attention to compare with reference
        X_combined = torch.cat([self_attended, X_ref.repeat(X.size(0), 1)], dim=1)
        trust_scores_cross = self.cross_attention(X_combined)
        
        # Additional classification score
        class_scores = self.classifier(X)
        
        # Combine both scores (weighted average)
        trust_scores = 0.7 * trust_scores_cross.squeeze() + 0.3 * class_scores.squeeze()
        
        return trust_scores 