import torch
from torch import nn
from torch_geometric.nn.inits import glorot


class NodePrompt(nn.Module):
    def __init__(self, size: int = 300, mode: str = 'add', enable_ssf: bool = False):
        super().__init__()
        # the initial value can be modified later
        self.enable_ssf = enable_ssf
        self.size = size
        self.value = torch.nn.Parameter(
            torch.randn(1, size)
        )
        self.b = torch.nn.Parameter(
            torch.randn(1, size)
        )
        self.init_weights()
        self.fixed_value = self.value.detach().clone()
        self.fixed_b = self.b.detach().clone()
        self.mode = mode

    def init_weights(self):
        nn.init.normal_(self.value, mean=1, std=0.02)
        nn.init.normal_(self.b, std=0.02)

    def forward(self, x: torch.Tensor):
        # the operation can be modified later
        if self.mode == 'add':
            return self._add(x)
        elif self.mode == 'mul':
            return self._mul(x)
        elif self.mode == 'mul_add':
            return self._mul_add(x)
        else:
            raise ValueError

    def _add(self, x: torch.Tensor):
        self.fixed_value = self.fixed_value.to(self.value.device)
        if self.enable_ssf:
            return x + self.value - self.fixed_value
        else:
            return x + self.value

    def _mul(self, x: torch.Tensor):
        return x * self.value

    def _mul_add(self, x: torch.Tensor):
        return x * self.value + self.b
