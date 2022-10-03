import random

import torch
import numpy as np

from src import config


def set_debug_mode():
    print('**********DEBUG MODE IS ON!**********')
    config.config_name = 'debug'
    config.Training.epochs = 1
    config.Training.batch_size = 32
    config.Logger.level = 'DEBUG'
    config.PretrainingDataLoader.num_workers = 0


def training_bar(epoch: int, idx: int, total: int, **kwargs):
    content = f'epoch {epoch + 1}:'
    for k, v in kwargs.items():
        content = ' '.join([content, f'[{k}:{v:.5f}]'])
    content = ' '.join([content, f'[progress:{(idx + 1) / total:0>6.2%}]'])
    return content


def worker_seed_init(idx: int, seed: int):
    torch_seed = torch.initial_seed()
    if torch_seed >= 2 ** 30:  # make sure torch_seed + worker_id < 2**32
        torch_seed = torch_seed % 2 ** 30
    seed = idx + seed + torch_seed
    random.seed(seed)
    np.random.seed(seed)


def set_seed(seed: int):
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
