import torch
import torch.nn as nn

class DualAttention(nn.Module):
    def __init__(self, feature_size):
        super(DualAttention, self).__init__()
        self.self_attention = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, feature_size)
        )
        self.cross_attention = nn.Sequential(
            nn.Linear(feature_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X, X_ref):
        self_attended = self.self_attention(X)
        X_combined = torch.cat([self_attended, X_ref.repeat(X.size(0), 1)], dim=1)
        trust_scores = self.cross_attention(X_combined)
        return trust_scores.squeeze() 