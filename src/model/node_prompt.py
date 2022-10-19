import torch
from torch import nn


class NodePrompt(nn.Module):
    def __init__(self, size: int = 300):
        super().__init__()
        # the initial value can be modified later
        self.size = size
        self.value = torch.nn.Parameter(
            torch.randn(1, size)
        )
        self.memory = self.value

    def remember(self):
        self.memory = self.value.clone()

    def reset(self):
        self.value = self.memory

    def forward(self, x: torch.Tensor):
        # the operation can be modified later
        return x + self.value
