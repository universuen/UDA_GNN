import torch
from torch import nn


class Prompt(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        # the initial value can be modified later
        self.value = torch.nn.Parameter(
            torch.randn(1, size)
        )

    def apply_to(self, x: torch.Tensor):
        # the operation can be modified later
        return x + self.value
