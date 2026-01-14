"""
attractiveness_model.py
-------------------------
PyTorch MLP regressor for attractiveness scoring.

Responsibilities:
- Define model architecture
- Forward pass
"""

import torch.nn as nn


class AttractivenessRegressor(nn.Module):
    def __init__(self, input_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)
