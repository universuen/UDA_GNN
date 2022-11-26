import torch
from torch import Tensor
from torch import nn
from torch.nn import functional


class SSLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.gamma, mean=1, std=.02)
        nn.init.normal_(self.beta, std=.02)
        self.fixed_gamma = self.gamma.detach().clone()
        self.fixed_beta = self.beta.detach().clone()

    def forward(self, x: Tensor) -> Tensor:
        y = functional.linear(x, self.weight, self.bias)
        self.fixed_gamma = self.fixed_gamma.to(self.gamma.device)
        self.fixed_beta = self.fixed_beta.to(self.beta.device)
        return self.gamma * y / self.fixed_gamma + self.beta - self.fixed_beta

    def reset_ss(self):
        self.gamma = nn.Parameter(torch.ones_like(self.bias))
        self.beta = nn.Parameter(torch.zeros_like(self.bias))
        nn.init.normal_(self.gamma, mean=1, std=.02)
        nn.init.normal_(self.beta, std=.02)
        self.fixed_gamma = self.gamma.detach().clone()
        self.fixed_beta = self.beta.detach().clone()
