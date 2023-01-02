from __future__ import annotations

import torch
from torch import nn


class SSFPrompt(nn.Module):
    def __init__(
            self,
            size: int = 300,
            uniform_init_interval: list[float, float] = None,
            batch_size: int = 32,
    ):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.ones(batch_size, size)
        )
        self.b = torch.nn.Parameter(
            torch.zeros(batch_size, size)
        )
        nn.init.normal_(self.w, mean=1, std=.02)
        nn.init.uniform_(self.b, *uniform_init_interval)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        return self.w[batch] * x + self.b[batch]
