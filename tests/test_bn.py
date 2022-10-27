import context

from torch import nn

from src import api

if __name__ == '__main__':
    api.replace_bn()
    api.set_bn_prior()
    bn = nn.BatchNorm1d(64)
    print(bn.prior)
    api.set_bn_prior(100)
    print(bn.prior)
