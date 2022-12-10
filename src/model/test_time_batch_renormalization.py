import torch
from torch import nn, Tensor


class TBR(nn.BatchNorm1d):
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.mu_ema = None
        self.theta_ema = None

        self.alpha = 0.9

    def forward(self, batch: Tensor) -> Tensor:
        mu_batch = torch.mean(batch.t(), dim=1).detach()
        theta_batch = torch.var(batch.t(), dim=1).detach()
        if self.mu_ema is None:
            self.mu_ema = mu_batch
        if self.theta_ema is None:
            self.theta_ema = theta_batch

        # calculate result
        r = mu_batch / self.mu_ema
        d = (mu_batch - self.mu_ema) / self.theta_ema
        batch = (batch - mu_batch) / theta_batch * r + d
        batch = self.weight * batch + self.bias

        # update statistics
        if self.training and self.track_running_stats:
            def update(x, y):
                return self.alpha * x + (1 - self.alpha) * y

            self.mu_ema = update(self.mu_ema, mu_batch)
            self.theta_ema = update(self.theta_ema, theta_batch)

        return batch

    @classmethod
    def replace_bn(cls):
        nn.BatchNorm1d = cls
