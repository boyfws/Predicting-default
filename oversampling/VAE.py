import random
import sys
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .DataTransformer import DataTransformer


class VAEWrapper:
    input_dim: int
    losses: NDArray

    def __init__(
        self,
        kl_weight: float = 1,
        latent_dim: int = 100,
        dims: tuple[int, ...] = (64, 128, 256, 256),
        dropout: Union[tuple[float, ...], float] = 0.0,
        lr: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 30,
        seed: int = 42,
        leaky_relu_coef: Union[tuple[float, ...]] = 0.2,
        device: torch.device | None = None,
        category_weight: float = 1.0,
        nan_weight: float = 1.0,
    ):
        if isinstance(dropout, tuple):
            if len(dims) > 2:
                assert len(dropout) == len(dims) - 2

            self.dropout = dropout

        else:
            self.dropout = tuple(dropout for _ in range(len(dims) - 2))

        if isinstance(leaky_relu_coef, tuple):
            assert len(leaky_relu_coef) == len(dims)

            self.leaky_relu_coef = leaky_relu_coef
        else:
            self.leaky_relu_coef = tuple(leaky_relu_coef for _ in range(len(dims)))

        self.data_transformer = DataTransformer()
        self.latent_dim = latent_dim
        self.dims = dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.kl_weight = kl_weight
        self.cross_entropy_weight = category_weight
        self.binary_cross_entropy_weight = nan_weight

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.fitted = False

    def _vae_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        recon_loss = self.data_transformer.calculate_mse_vae(
            recon_x,
            x,
            cross_entropy_weight=self.cross_entropy_weight,
            binary_cross_entropy_weight=self.binary_cross_entropy_weight,
        )

        recon_loss = recon_loss / x.size(0)

        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )  # [B]
        kl_loss = torch.mean(kl_per_sample)

        return recon_loss + self.kl_weight * kl_loss

    def _check_fit(self):
        if not self.fitted:
            raise RuntimeError("Instance has not been fitted yet.")

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, input_dim: int) -> None:
        dims = self.dims
        leaky = self.leaky_relu_coef
        dropout = self.dropout

        encoder_layers = (
            [
                nn.Linear(input_dim, dims[0]),
                nn.LayerNorm(dims[0]),
                nn.LeakyReLU(leaky[0]),
            ]
            + [
                layer
                for i in range(len(dims) - 1)
                for layer in (
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    nn.LeakyReLU(leaky[i + 1]),
                    nn.Dropout(dropout[i]) if i < len(dropout) else nn.Identity(),
                )
            ]
            + [nn.Linear(dims[-1], self.latent_dim * 2)]
        )

        decoder_layers = (
            [
                nn.Linear(self.latent_dim, dims[-1]),
                nn.LayerNorm(dims[-1]),
                nn.LeakyReLU(leaky[-1]),
            ]
            + [
                layer
                for i in range(len(dims) - 1, 0, -1)
                for layer in (
                    nn.Linear(dims[i], dims[i - 1]),
                    nn.LayerNorm(dims[i - 1]),
                    nn.LeakyReLU(leaky[i - 1]),
                    (
                        nn.Dropout(dropout[i - 2])
                        if (i - 2) >= 0 and (i - 2) < len(dropout)
                        else nn.Identity()
                    ),
                )
            ]
            + [nn.Linear(dims[0], input_dim), nn.Tanh()]
        )

        class VAE(nn.Module):
            def __init__(self):
                super().__init__()

                self.encoder = nn.Sequential(*encoder_layers)
                self.decoder = nn.Sequential(*decoder_layers)

            def reparameterize(
                self, mu: torch.Tensor, logvar: torch.Tensor
            ) -> torch.Tensor:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(
                self, X: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                mu, logvar = self.encoder(X).chunk(2, dim=1)
                z = self.reparameterize(mu, logvar)
                x_recon = self.decoder(z)
                return x_recon, mu, logvar

            def to_latent(self, X: torch.Tensor) -> torch.Tensor:
                mu, logvar = self.encoder(X).chunk(2, dim=1)
                z = self.reparameterize(mu, logvar)
                return z

            def from_latent(self, z: torch.Tensor) -> torch.Tensor:
                return self.decoder(z)

        self.model = VAE().to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(
        self, df: pd.DataFrame, num_workers: int = 4, prefetch_factor: int = 20
    ) -> "VAEWrapper":

        self._set_seed(self.seed)

        data = self.data_transformer.fit_transform(df).float()
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=prefetch_factor,
        )
        input_dim = data.size(1)

        self.input_dim = input_dim
        self._build_model(input_dim)
        self.model.train()

        self.losses = np.zeros((self.epochs,), dtype=np.float64)

        epoch_len = len(str(self.epochs))
        for epoch in range(self.epochs):
            epoch_loss = []
            pbar = tqdm(
                loader,
                desc=f"Epoch {str(epoch + 1).rjust(epoch_len)}/{self.epochs}",
                file=sys.stdout,
                leave=True,
            )
            for i, (real_batch,) in enumerate(pbar):
                real_batch = real_batch.to(self.device)  # noqa: PLW2901

                self.optim.zero_grad()

                recon_batch, mu, logvar = self.model(real_batch)

                loss = self._vae_loss(recon_batch, real_batch, mu, logvar)

                loss.backward()
                self.optim.step()

                epoch_loss.append(loss.item())

                if i != len(loader) - 1:
                    dict_to_print = {
                        "Loss": f"{loss.item():.4f}",
                    }

                    pbar.set_postfix(dict_to_print)
                else:
                    self.losses[epoch] = np.mean(epoch_loss)

                    pbar.set_postfix(
                        {
                            "Av. loss": f"{self.losses[epoch]:.4f}",
                        }
                    )

        self.model.eval()

        self.fitted = True
        return self

    def to_latent(self, X: pd.DataFrame, seed: Optional[int] = None) -> torch.Tensor:
        self._check_fit()

        if seed is not None:
            self._set_seed(seed)

        self.model.eval()

        with torch.no_grad():

            return self.model.to_latent(
                self.data_transformer.transform(X).to(self.device),
            ).cpu()

    def from_latent(
        self,
        X: torch.Tensor,
    ) -> pd.DataFrame:
        self._check_fit()

        self.model.eval()

        with torch.no_grad():
            return self.data_transformer.inverse_transform(
                self.model.from_latent(X.to(self.device)).cpu(),
            )

    def save(self, filepath: str | Path) -> None:
        self._check_fit()
        self.model.eval()

        data_transformer_params = self.data_transformer.get_params()
        state_dict = self.model.state_dict()

        final_dict = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optim.state_dict(),
            "data_transformer_params": data_transformer_params,
            "model_params": [
                list(self.dims),
                list(self.leaky_relu_coef),
                list(self.dropout),
                self.latent_dim,
                self.input_dim,
            ],
        }
        torch.save(final_dict, filepath)

    def load(self, filepath: str | Path) -> "VAEWrapper":

        data = torch.load(filepath, weights_only=False)

        self.data_transformer = DataTransformer().save_params(
            data["data_transformer_params"]
        )

        model_params = data["model_params"]
        self.dims = tuple(model_params[0])
        self.leaky_relu_coef = tuple(model_params[1])
        self.dropout = tuple(model_params[2])
        self.latent_dim = model_params[3]
        self.input_dim = model_params[4]
        old_device = self.device

        self.device = torch.device("cpu")
        self._build_model(self.input_dim)
        self.model.load_state_dict(data["model_state_dict"])
        self.optim.load_state_dict(data["optimizer_state_dict"])

        self.device = old_device
        self.model.to(self.device)
        self.model.eval()

        self.fitted = True

        return self

    def plot_fit(self, figsize: tuple[int, int] = (10, 10)) -> None:
        self._check_fit()

        mean_losses = self.losses
        epochs = np.arange(1, mean_losses.shape[0] + 1)

        plt.figure(figsize=figsize)
        plt.plot(epochs, mean_losses, label="D_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()
