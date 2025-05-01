import torch
import torch.nn as nn


class NoiseInjection(nn.Module):
    def __init__(self, features_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(features_dim))

    def forward(self, x):
        batch_size, dim = x.shape
        noise = torch.randn(batch_size, dim, device=x.device)
        return x + noise * self.weight