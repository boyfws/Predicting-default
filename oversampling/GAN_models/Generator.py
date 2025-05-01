import torch
import torch.nn as nn

from .Noise_injector import NoiseInjection


class Generator(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dims: tuple[int, int, int, int],
                 output_dim: int
                 ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            NoiseInjection(hidden_dims[1]),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[3], output_dim),

            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)