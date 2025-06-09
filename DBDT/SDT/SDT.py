import torch
from torch import nn

from .BaseSplitter import BaseSplitter


class SDT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        regularization: bool = True,
        t: float = 1,
    ) -> None:
        super().__init__()
        assert depth >= 1

        self.depth = depth
        self.regularization = regularization

        self.splitter = BaseSplitter(input_dim, t, depth)

        self.value = nn.Parameter(torch.empty(2**depth, output_dim))
        nn.init.xavier_uniform_(self.value)

        d = torch.arange(self.depth).repeat_interleave(2 ** torch.arange(self.depth))
        self.register_buffer("d", d)

    def forward(self, x, left_mask, right_mask):
        predicted_probs = self.splitter(x)  # [Batch_size, 2 ** depth - 1]

        if self.regularization:
            reg_term = (
                -0.5
                * (
                    0.5**self.d
                    * torch.log(
                        torch.clamp(predicted_probs * (1 - predicted_probs), min=1e-5)
                    )
                ).mean()
            )

        else:
            reg_term = None

        log_p = torch.log(torch.clamp(predicted_probs, min=1e-5))  # [B, N]
        log_1mp = torch.log(torch.clamp(1 - predicted_probs, min=1e-5))  # [B, N]

        # left_mask  [L, N]
        # right_mask  [L, N]

        accum_probs = log_p @ left_mask + log_1mp @ right_mask  # [B, L]
        accum_probs = accum_probs.unsqueeze(-1)  # [B, L, 1]

        sign = torch.sign(self.value)  # [L, D]
        logval = torch.log(torch.clamp(self.value.abs(), min=1e-10))  # [L, D]

        logval = logval.unsqueeze(0)  # [1, L, D]
        ret = accum_probs + logval  # [B, L, D]

        pos_mask = (sign > 0).unsqueeze(0)  # [1, L, D]
        neg_mask = (sign < 0).unsqueeze(0)  # [1, L, D]

        ret_pos = ret.masked_fill(~pos_mask, -float("inf"))  # [B, L, D]
        ret_neg = ret.masked_fill(~neg_mask, -float("inf"))  # [B, L, D]

        log_pos = torch.logsumexp(ret_pos, dim=1)  # [B, D]
        log_neg = torch.logsumexp(ret_neg, dim=1)  # [B, D]

        value = torch.exp(log_pos) - torch.exp(log_neg)  # [B, D]

        return value, reg_term

    def eval(self):
        self.regularization = False
        super().eval()
