from copy import deepcopy

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional


class SSLinear(nn.Linear):
    ss_enabled: bool = False

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.gamma = torch.randn(self.weight.shape)
        self.beta = torch.randn(self.bias.shape)
        self.fixed_gamma = deepcopy(self.gamma).detach()
        self.fixed_beta = deepcopy(self.beta).detach()

    def forward(self, x: Tensor) -> Tensor:
        y = functional.linear(x, self.weight, self.bias)
        if not SSLinear.ss_enabled:
            return y
        else:
            return self.gamma * y / self.fixed_gamma + self.beta - self.fixed_beta
