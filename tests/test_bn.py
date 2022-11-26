import torch

import context

from torch import nn

from src import api

if __name__ == '__main__':
    api.replace_bn()
    api.set_bn_prior(0.5)
    bn = nn.BatchNorm1d(64)
    print(bn.prior)
    api.set_bn_prior(100)
    print(bn.prior)
    bn(torch.randn(10, 64))
