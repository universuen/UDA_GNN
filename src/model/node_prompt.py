import torch
from torch import nn
from torch_geometric.nn.inits import glorot


class NodePrompt(nn.Module):
    def __init__(self, size: int = 300):
        super().__init__()
        # the initial value can be modified later
        self.size = size
        self.value = torch.nn.Parameter(
            torch.randn(1, size)
        )
        glorot(self.value)
        self.memory = None
        self.remember()

    def remember(self):
        self.memory = self.value.data.detach().clone()

    def reset(self):
        self.value.data.copy_(self.memory)

    def forward(self, x: torch.Tensor):
        # the operation can be modified later
        return x + self.value
