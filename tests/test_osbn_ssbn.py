import torch

import src


if __name__ == '__main__':
    src.api.replace_bn()
    src.api.replace_with_ssf()

    bn = torch.nn.BatchNorm1d(10)
    print(bn(torch.randn(3, 10)))
