import math

import torch
from torch import nn


class VampPrior(nn.Module):
    def __init__(self, encoder_net, n_components, latent_dim):
        super().__init__()
        self.encoder_net = encoder_net
        self.n_components = n_components
        self.latent_dim = latent_dim

        self.means = nn.Parameter(torch.randn(n_components, latent_dim))
        self.logvars = nn.Parameter(torch.zeros(n_components, latent_dim))

    def forward(self, z):
        z = z.unsqueeze(1)  # [B, 1, D]
        mean = self.means.unsqueeze(0)  # [1, K, D]
        logvar = self.logvars.unsqueeze(0)  # [1, K, D]

        log_probs = -0.5 * (logvar + ((z - mean) ** 2) / logvar.exp())
        log_probs = log_probs.sum(dim=2)  # [B, K]

        log_prior = torch.logsumexp(log_probs - math.log(self.n_components), dim=1)
        return log_prior


class EnhancedVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        dims,
        dropout,
        leaky_relu_coef,
        use_vampprior=False,
        n_pseudo_inputs=500,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_vampprior = use_vampprior

        encoder_layers = []
        prev_dim = input_dim
        for i, h in enumerate(dims):
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.LayerNorm(h))
            encoder_layers.append(nn.LeakyReLU(leaky_relu_coef[i]))
            if i < len(dropout):
                encoder_layers.append(nn.Dropout(dropout[i]))
            prev_dim = h
        self.encoder_net = nn.Sequential(*encoder_layers)
        self.encoder_out = nn.Linear(prev_dim, 2 * latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for i in reversed(range(len(dims))):
            decoder_layers.append(nn.Linear(prev_dim, dims[i]))
            decoder_layers.append(nn.LayerNorm(dims[i]))
            decoder_layers.append(nn.LeakyReLU(leaky_relu_coef[i]))
            if i - 1 >= 0 and i - 1 < len(dropout):
                decoder_layers.append(nn.Dropout(dropout[i - 1]))
            prev_dim = dims[i]
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

        if use_vampprior:
            self.vamp = VampPrior(self.encoder_net, n_pseudo_inputs, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder_net(x)
        mu, logvar = self.encoder_out(h).chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def compute_kl(self, mu, logvar, z):
        if self.use_vampprior:
            # лог-плотность q(z|x)
            log_qz = -0.5 * torch.sum(
                ((z - mu) ** 2) / logvar.exp() + logvar + math.log(2 * math.pi), dim=1
            )
            #  p(z) ~ VampPrior
            log_pz = self.vamp(z)
            return torch.mean(log_qz - log_pz)
        else:
            return torch.mean(
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

    def to_latent(self, x):
        h = self.encoder_net(x)
        mu, logvar = self.encoder_out(h).chunk(2, dim=1)
        return self.reparameterize(mu, logvar)

    def from_latent(self, z):
        return self.decoder(z)
