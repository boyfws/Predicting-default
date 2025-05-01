import sys

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
import random
import pandas as pd

from .DataTransformer import DataTransformer
from .GAN_models import *

from typing import Optional
from pathlib import Path


class OversampleGAN:
    def __init__(
        self,
        latent_dim: int = 100,
        hidden_dims: tuple[int, int, int, int] = (64, 128, 256, 256),
        dropout: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        D_lr: float = 1e-4,
        G_lr: float = 4e-4,
        batch_size: int = 512,
        epochs: int = 30,
        seed: int = 42,
        leaky_relu_coef: float = 0.2,
        device: torch.device | None = None,
        n_critic: int = 5
    ):
        self.data_transformer = DataTransformer()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.D_lr = D_lr
        self.G_lr = G_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.leaky_relu_coef = leaky_relu_coef
        self.seed = seed
        self.dropout = dropout
        self.n_critic = n_critic


        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.fitted = False

    def _check_fit(self):
        if not self.fitted:
            raise RuntimeError('Instance has not been fitted yet.')

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_models(self, input_dim: int):
        self.G = Generator(self.latent_dim, self.hidden_dims, input_dim).to(self.device)
        self.D = Discriminator(input_dim, self.hidden_dims[::-1], self.leaky_relu_coef, self.dropout).to(self.device)
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(0.0, 0.999))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(0.0, 0.999))

    def _fit_D(
            self,
            real_batch: torch.Tensor,
            fake_batch: torch.Tensor,
    ) -> float:
        self.optim_D.zero_grad()

        out_real = self.D(real_batch)
        out_fake = self.D(fake_batch)

        loss_D = out_fake.mean() - out_real.mean()

        loss_D.backward()
        self.optim_D.step()

        return loss_D.item()

    def _fit_G(
        self,
        fake_batch: torch.Tensor,

    ) -> float:
        self.optim_G.zero_grad()
        self.optim_D.zero_grad()

        out = self.D(fake_batch)
        loss_G = - out.mean()
        loss_G.backward()
        self.optim_G.step()

        return loss_G.item()

    def fit(self,
            df: pd.DataFrame,
            num_workers: int = 4,
            prefetch_factor: int = 20
        ) -> "OversampleGAN":
        self._set_seed(self.seed)

        data = self.data_transformer.fit_transform(df).float()
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=prefetch_factor
        )

        input_dim = data.size(1)

        self.input_dim = input_dim
        self._build_models(input_dim)

        self.G.train()
        self.D.train()

        self.losses = np.zeros(
            (self.epochs, 2), dtype=np.float64
        )

        for epoch in range(self.epochs):
            pbar = tqdm(
                    loader,
                    desc=f"Epoch {str(epoch + 1).rjust(len(str(self.epochs)))}/{self.epochs}",
                    file=sys.stdout,
                    leave=True
            )
            epoch_loss_D = []
            epoch_loss_G = []
            for i, (real_batch,) in enumerate(pbar):
                real_batch = real_batch.to(self.device)
                bs = real_batch.size(0)

                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_batch = self.G(z)

                loss_D = self._fit_D(
                        real_batch,
                        fake_batch.detach(),
                )

                if i % self.n_critic == 0:

                    loss_G = self._fit_G(
                            fake_batch,
                    )
                    epoch_loss_G.append(loss_G)

                epoch_loss_D.append(loss_D)

                if i != len(loader) - 1:
                    dict_to_print = {
                            "D_loss": f"{loss_D:.4f}",
                    }
                    if i % self.n_critic == 0:
                        dict_to_print["G_loss"] = f"{loss_G:.4f}"

                    pbar.set_postfix(dict_to_print)
                else:
                    self.losses[epoch, 0] = np.mean(epoch_loss_D)
                    self.losses[epoch, 1] = np.mean(epoch_loss_G)

                    pbar.set_postfix({
                            "D_loss": f"{self.losses[epoch, 0]:.4f}",
                            "G_loss": f"{self.losses[epoch, 1]:.4f}"
                    })

        self.G.eval()
        self.D.eval()

        self.fitted = True
        return self

    def generate(
            self,
            num_samples: int,
            seed: Optional[int] = None
    ) -> pd.DataFrame:
        self._check_fit()

        if seed is not None:
            self._set_seed(seed)

        self.G.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            gen_tensor = self.G(z).cpu()

        return self.data_transformer.inverse_transform(gen_tensor)

    def save_generator(
            self,
            filepath: str | Path
    ) -> None:
        self._check_fit()

        if not hasattr(self, 'G') or self.G is None:
            raise RuntimeError("Model is not fitted")

        data_transformer_params = self.data_transformer.get_params()
        state_dict = self.G.state_dict()

        final_dict = {
            "model": state_dict,
            "data_transformer_params": data_transformer_params,
            "G_params": [self.latent_dim, list(self.hidden_dims), self.input_dim]
        }
        torch.save(final_dict, filepath)

    def load_generator(
            self,
            filepath: str | Path
    ) -> "OversampleGAN":
        data = torch.load(filepath, weights_only=False)

        self.data_transformer = DataTransformer().save_params(
            data["data_transformer_params"]
        )
        G_params = data["G_params"]

        self.latent_dim, self.hidden_dims, self.input_dim = G_params[0], tuple(G_params[1]), G_params[2]
        self.G = Generator(*G_params).to(self.device)
        self.G.load_state_dict(data["model"])
        self.G.eval()

        self.fitted = True

        return self

    def plot_fit(
            self,
            figsize: tuple[int, int] = (10, 10)
    ) -> None:
        self._check_fit()

        self.G.eval()
        mean_losses = self.losses
        epochs = np.arange(1, mean_losses.shape[0] + 1)

        plt.figure(figsize=figsize)
        plt.plot(epochs, mean_losses[:, 0], label="D_loss")
        plt.plot(epochs, mean_losses[:, 1], label="G_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()