import torch
import torch.nn as nn
from torch import autograd

class Discriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: tuple[int, int, int, int],
                 leaky_relu_coef: float,
                 dropout: tuple[float, float, float, float]
                 ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(leaky_relu_coef, inplace=True),
            nn.Dropout(dropout[0]) if dropout[0] > 0 else nn.Identity(),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(leaky_relu_coef, inplace=True),
            nn.Dropout(dropout[1]) if dropout[1] > 0 else nn.Identity(),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(leaky_relu_coef, inplace=True),
            nn.Dropout(dropout[2]) if dropout[2] > 0 else nn.Identity(),

            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.LeakyReLU(leaky_relu_coef, inplace=True),
            nn.Dropout(dropout[3]) if dropout[3] > 0 else nn.Identity(),

            nn.Linear(hidden_dims[3], 1),
            nn.Sigmoid()
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
        x = x.requires_grad_()

        out = self.net(x)  # shape (batch, 1)

        grad = autograd.grad(
            outputs=out.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]

        grad_norm = grad.view(x.size(0), -1).norm(2, dim=1, keepdim=True)

        return (1 - out) / (grad_norm + (1 - out).abs() + 1e-6)
