import sys

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import random
import pandas as pd

from .DataTransformer import DataTransformer


class OversampleGAN:
    def __init__(
        self,
        latent_dim: int = 100,
        hidden_dims: tuple[int, int] = (128, 64),
        D_lr: float = 1e-4,
        G_lr: float = 4e-4,
        batch_size: int = 512,
        epochs: int = 30,
        seed: int = 42,
        leaky_relu_coef: float = 0.2,
        device: torch.device | None = None,
    ):
        self.data_transformer = DataTransformer()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.D_lr = D_lr
        self.G_lr = G_lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.leaky_relu_coef = leaky_relu_coef

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def _build_models(self, input_dim: int):

        class Generator(nn.Module):
            def __init__(self, latent_dim, hidden_dims, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.BatchNorm1d(hidden_dims[1]),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dims[1], output_dim),
                )
                self._init_weights()

            def _init_weights(self):
                for m in self.net:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)

            def forward(self, z):
                return self.net(z)

        leaky_relu_coef = self.leaky_relu_coef

        class Discriminator(nn.Module):
            def __init__(self, input_dim, hidden_dims):
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


        self.G = Generator(self.latent_dim, self.hidden_dims, input_dim).to(self.device)
        self.D = Discriminator(input_dim, self.hidden_dims).to(self.device)
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(0.5, 0.999))

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

        data = self.data_transformer.fit_transform(df).float()
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=str(self.device).lower() == "cuda",
            prefetch_factor=prefetch_factor
        )

        input_dim = data.size(1)
        self._build_models(input_dim)

        self.G.train()
        self.D.train()

        self.losses = np.zeros(
            (self.epochs, len(loader), 2), dtype=np.float64
        )

        for epoch in range(self.epochs):
            pbar = tqdm(
                    loader,
                    desc=f"Epoch {str(epoch + 1).rjust(len(str(self.epochs)))}/{self.epochs}",
                    file=sys.stdout,
                    leave=True
            )
            for i, (real_batch,) in enumerate(pbar):
                real_batch = real_batch.to(self.device)
                bs = real_batch.size(0)

                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake_batch = self.G(z)

                loss_D = self._fit_D(
                        real_batch,
                        fake_batch.detach(),
                )

                loss_G = self._fit_G(
                        fake_batch,
                )

                self.losses[epoch, i, 0] = loss_D
                self.losses[epoch, i, 1] = loss_G

                if i != len(loader) - 1:
                    pbar.set_postfix({
                            "D_loss": f"{loss_D:.4f}",
                            "G_loss": f"{loss_G:.4f}"
                    })
                else:
                    pbar.set_postfix({
                            "D_loss": f"{np.mean(self.losses[epoch, :, 0]):.4f}",
                            "G_loss": f"{np.mean(self.losses[epoch, :, 1]):.4f}"
                    })

        self.G.eval()
        self.D.eval()

        return self

    def generate(self, num_samples: int) -> pd.DataFrame:
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            gen_tensor = self.G(z).cpu()
        return self.data_transformer.inverse_transform(gen_tensor)

    def save_generator(
            self,
            filepath: str
    ) -> None:
        if not hasattr(self, 'G') or self.G is None:
            raise RuntimeError("Model is not fitted")
        torch.save(self.G.state_dict(), filepath)

    def load_generator(
            self,
            filepath: str
    ) -> "OversampleGAN":
        pass

    def plot_fit(self, figsize: tuple[int, int] = (10, 10)) -> None:
        mean_losses = self.losses.mean(axis=1)
        epochs = np.arange(1, mean_losses.shape[0] + 1)

        plt.figure(figsize=figsize)
        plt.plot(epochs, mean_losses[:, 0], label="D_loss")
        plt.plot(epochs, mean_losses[:, 1], label="G_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()