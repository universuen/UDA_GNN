import logging
import random
from pathlib import Path
from typing import Callable
from typing import Type

import torch
import numpy as np

datasets = [
    "bace",
    "sider",
    "bbbp",
    "tox21",
    "clintox",
    "muv",
    "toxcast",
    "hiv"
]


class _Config:
    @classmethod
    def to_dict(cls) -> dict:
        result = dict()
        for k, v in vars(cls).items():
            if k.startswith('_') or k in ('i', 'j', 'k'):
                continue
            result[k] = v
        return result

    @classmethod
    def print_content(cls):
        print(cls.__name__)
        for k, v in cls.to_dict().items():
            print(f'\t{k}: {v}')


config_name: str | None = None
device: str = 'cuda:0'
seed: int = 0

ConfigType = Type[_Config]


class Paths(_Config):
    project: Path = Path(__file__).absolute().parent
    src: Path = project / 'src'
    data: Path = project / 'data'
    scripts: Path = project / 'scripts'
    tests: Path = project / 'tests'
    logs: Path | None = data / 'logs'
    datasets: Path = data / 'datasets'
    results: Path = data / 'results'
    models: Path = data / 'models'

    # create path if not exists
    for i in list(vars().values()):
        if isinstance(i, Path):
            i.mkdir(parents=True, exist_ok=True)


class Pretraining(_Config):
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 2048


class Tuning(_Config):
    epochs: int = 100
    lr: float = 1e-4
    batch_size: int = 128


class PretrainingDataset(_Config):
    dataset: str = 'zinc_standard_agent'
    aug_1: str = 'random'
    aug_ratio_1: int | float = 0.2
    aug_2: str = 'random'
    aug_ratio_2: int | float = 0.2
    use_original: bool = False

    # validity check
    assert aug_1 in (
        'dropN',
        'permE',
        'maskN',
        'subgraph',
        'random',
        'random_v2',
        'none',
    )
    assert 0 <= aug_ratio_1 <= 1
    assert aug_2 in (
        'dropN',
        'permE',
        'maskN',
        'subgraph',
        'random',
        'random_v2',
        'none',
    )
    assert 0 <= aug_ratio_2 <= 1


class TuningDataset(_Config):
    dataset: str = 'clintox'


def _worker_seed_init(idx: int, seed_: int):
    torch_seed = torch.initial_seed()
    if torch_seed >= 2 ** 30:  # make sure torch_seed + worker_id < 2**32
        torch_seed = torch_seed % 2 ** 30
    seed_ = idx + seed_ + torch_seed
    random.seed(seed_)
    np.random.seed(seed_)


class PretrainingLoader(_Config):
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    worker_init_fn: Callable = lambda x: _worker_seed_init(x, seed)


class TuningLoader(_Config):
    num_workers: int = 4
    pin_memory: bool = True


class EqvGNN(_Config):
    num_layer: int = 5
    emb_dim: int = 300
    use_edge_dropout: bool = False
    jk: str = "last"
    drop_ratio: int | float = 0

    # validity check
    assert jk in ("concat", "last", "max", "sum")
    assert 0 <= drop_ratio <= 1


class BarlowTwins(_Config):
    lambda_: float = 0.0051
    sizes: tuple[int] = (300, 1200, 1200, 1200)


class Logger(_Config):
    message_fmt: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    level: int | str = logging.INFO
    date_fmt: str = '%Y-%m-%d %H:%M:%S'
