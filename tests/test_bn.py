import torch

import context

from torch import nn

from src import api

if __name__ == '__main__':
    t = torch.randn(2, 10)
    bn = nn.BatchNorm1d(10)
    print(bn(t))
    api.replace_bn()
    # t = torch.randn(2, 10)
    bn = nn.BatchNorm1d(10)
    print(bn(t))
