import torch
from torch.nn import BatchNorm1d, functional


class OneSampleBN(BatchNorm1d):
    prior: float = 1

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if abs(OneSampleBN.prior - 1) <= 1e-5:
            return super().forward(input_)
        est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
        est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
        functional.batch_norm(input_, est_mean, est_var, None, None, True, 1.0, self.eps)
        running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
        running_var = self.prior * self.running_var + (1 - self.prior) * est_var
        return functional.batch_norm(input_, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)
