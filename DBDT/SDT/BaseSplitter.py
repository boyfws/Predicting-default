import torch.nn.functional as F
from torch import nn


class BaseSplitter(nn.Module):
    def __init__(self, input_dim: int, t: float, depth: int):
        super().__init__()
        self.t = t

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10 * (2**depth - 1)),
            nn.GELU(),
            nn.Linear(10 * (2**depth - 1), 2**depth - 1),
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.t * x

        return F.sigmoid(x)  # [Batch_size, Output_dim]
