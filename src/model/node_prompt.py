import torch
from torch import nn
from torch_geometric.nn.inits import glorot


class NodePrompt(nn.Module):
    def __init__(
            self,
            size: int = 300,
            mode: str = 'add',
            enable_ssf: bool = False,
            uniform_init_interval: list[int, int] = None,
    ):
        super().__init__()

        self.enable_ssf = enable_ssf
        self.size = size

        self.value = torch.nn.Parameter(
            torch.randn(1, size)
        )
        self.b = torch.nn.Parameter(
            torch.randn(1, size)
        )

        self.init_weights()
        if uniform_init_interval is not None:
            nn.init.uniform_(self.b, *uniform_init_interval)

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
            return x + self.b - self.fixed_b
        else:
            return x + self.b

    def _mul(self, x: torch.Tensor):
        return x * self.value

    def _mul_add(self, x: torch.Tensor):
        return x * self.value + self.b


class NodePromptPtb(nn.Module):
    def __init__(
            self,
            size: int = 300,
            uniform_init_interval: list[int, int] = None,
            batch_size: int = 32,
    ):
        super().__init__()
        self.b = torch.nn.Parameter(
            torch.zeros(batch_size, size)
        )
        nn.init.uniform_(self.b, *uniform_init_interval)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        '''
        x: shape = [N, D]
        batch: shape = [N]
        '''
        return x + self.b[batch]


class NodePrompt_v2(nn.Module):
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

    def to(self, args):
        super().to(args)
        self.fixed_value = self.fixed_value.to(args)
        self.fixed_b = self.fixed_b.to(args)
        return self

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
        elif self.mode == 'batch_add':
            return self._batch_add(x)
        else:
            raise ValueError
    
    def _batch_add(self, x):
        '''
        x: shape = [N, D]
        batch: shape = [N]
        '''
        N = x.shape[0]
        if not hasattr(self, 'batch_b'):
            batch_b = self.b.view(1, -1).repeat(N, 1) # shape = [N, D]
            batch_b.retain_grad()
            self.batch_b = batch_b
            self.batch_b2 = batch_b.detach().clone()
        else:
            batch_b = self.batch_b
        x = x + batch_b
        return x
    
    def step(self, lr):
        self.batch_b = self.batch_b - lr * self.batch_b.grad

    def reset(self):
        if hasattr(self, 'batch_b'):
            delattr(self, 'batch_b')

    def _add(self, x: torch.Tensor):
        self.fixed_value = self.fixed_value.to(self.value.device)
        if self.enable_ssf:
            return x + self.b - self.fixed_b
        else:
            return x + self.b

    def _mul(self, x: torch.Tensor):
        return x * self.value

    def _mul_add(self, x: torch.Tensor):
        return x * self.value + self.b
