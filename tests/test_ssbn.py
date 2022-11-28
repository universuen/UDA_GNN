import context

import torch

import src

if __name__ == '__main__':
    torch.nn.BatchNorm1d = src.model.SSBatchNorm
    ss_bn = torch.nn.BatchNorm1d(50)
    print(ss_bn.gamma)
    print(ss_bn.beta)
    print(ss_bn(torch.randn(10, 50)))
