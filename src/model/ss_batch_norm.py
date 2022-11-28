import torch

from torch import nn, Tensor

from .one_sample_bn import OneSampleBN


class SSBatchNorm(OneSampleBN):

    def __init__(self, num_features):
        super().__init__(num_features)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        nn.init.normal_(self.gamma, mean=1, std=.02)
        nn.init.normal_(self.beta, std=.02)
        self.fixed_gamma = self.gamma.detach().clone()
        self.fixed_beta = self.beta.detach().clone()

    def forward(self, x: Tensor) -> Tensor:
        y = super().forward(x)
        self.fixed_gamma = self.fixed_gamma.to(self.gamma.device)
        self.fixed_beta = self.fixed_beta.to(self.beta.device)
        return self.gamma * y / self.fixed_gamma + self.beta - self.fixed_beta
