import torch
from torch import nn, Tensor


class TBR(nn.BatchNorm1d):
    enable = False
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.register_buffer('running_std', torch.zeros(num_features, device=self.running_var.device))
        # self.running_std = nn.Parameter(torch.zeros_like(self.running_var))
        self.running_std = self.running_var ** 0.5
        self.alpha = 0.8
        self.to(self.running_var.data.device)

    def forward(self, batch: Tensor) -> Tensor:
        if not TBR.enable:
            return super().forward(batch)
        mean_batch = torch.mean(batch.t(), dim=1).detach()
        var_batch = torch.var(batch.t(), dim=1).detach()
        # update statistics
        if self.training and self.track_running_stats:
            def update(x, y):
                return self.alpha * x + (1 - self.alpha) * y
            self.running_mean = update(self.running_mean, mean_batch.detach())
            self.running_var = update(self.running_var, var_batch.detach())

        # calculate result
        r = (var_batch.detach() ** 0.5) / (self.running_var ** 0.5)
        d = (mean_batch.detach() - self.running_mean) / (self.running_var ** 0.5)
        batch = (batch - mean_batch) / (var_batch ** 0.5) * r + d
        batch = self.weight * batch + self.bias
        return batch

    @classmethod
    def replace_bn(cls):
        nn.BatchNorm1d = cls
