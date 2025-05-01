import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: tuple[int, int],
                 leaky_relu_coef: float,
                 dropout: tuple[float, float]
                 ):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dims[0]), n_power_iterations=1),
            nn.LeakyReLU(leaky_relu_coef, inplace=True)
        ]
        if dropout[0] >= 1e-7:
            layers += [
                nn.Dropout(dropout[0])
            ]

        layers += [
            nn.utils.spectral_norm(nn.Linear(hidden_dims[0], hidden_dims[1]), n_power_iterations=1),
            nn.LeakyReLU(leaky_relu_coef, inplace=True),
        ]

        if dropout[1] >= 1e-7:
            layers += [nn.Dropout(dropout[1])]

        layers += [
            nn.utils.spectral_norm(nn.Linear(hidden_dims[1], 1), n_power_iterations=1)
        ]

        self.net = nn.Sequential(*layers)
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