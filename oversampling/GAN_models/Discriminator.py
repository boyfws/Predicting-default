import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, leaky_relu_coef):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dims[1]), n_power_iterations=1),
            nn.LeakyReLU(leaky_relu_coef, inplace=True),

            nn.utils.spectral_norm(nn.Linear(hidden_dims[1], hidden_dims[0]), n_power_iterations=1),
            nn.LeakyReLU(leaky_relu_coef, inplace=True),

            nn.utils.spectral_norm(nn.Linear(hidden_dims[0], 1), n_power_iterations=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif hasattr(m, "weight_orig"):
                nn.init.xavier_uniform_(m.weight_orig)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)