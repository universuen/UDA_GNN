import torch
from torch import nn
from torch_geometric.nn.inits import glorot


class NodePrompt(nn.Module):
    def __init__(self, size: int = 300, mode: str = 'add'):
        super().__init__()
        # the initial value can be modified later
        self.size = size
        self.value = torch.nn.Parameter(
            torch.randn(1, size)
        )
        self.b = torch.nn.Parameter(
            torch.randn(1, size)
        )
        self.init_weights()
        self.memory = {
            'value': self.value.clone(),
            'b': self.b.clone(),
        }
        self.remember()
        self.mode = mode

    def init_weights(self):
        glorot(self.value)
        glorot(self.b)

    def remember(self):
        self.memory['value'] = self.value.clone()
        self.memory['b'] = self.b.clone()

    def reset(self):
        self.value = self.memory['value'].clone()
        self.b = self.memory['b'].clone()

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
        return x + self.value

    def _mul(self, x: torch.Tensor):
        return x * self.value

    def _mul_add(self, x: torch.Tensor):
        return x * self.value + self.b
