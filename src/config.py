from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Type

import torch
import numpy as np

import src.dataset

datasets = [
    "bbbp",
    "tox21",
    "toxcast",
    "sider",
    "clintox",
    "muv",
    "hiv",
    "bace",
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
loop_seeds: list[int] = [i for i in range(10)]

ConfigType = Type[_Config]


class Paths(_Config):
    src: Path = Path(__file__).absolute().parent
    project: Path = src.parent
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
    use_graph_trans: bool = False
    use_dual_dataset: bool = False
    dual_ds_version: int = 1
    save_epoch: int = 20


class Tuning(_Config):
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    use_lr_scheduler: bool = True
    use_node_prompt: bool = False
    use_edge_prompt: bool = False


class TestTimeTuning(_Config):
    num_augmentations: int = 32
    num_iterations: int = 1
    aug: str = 'random'
    aug_ratio: int | float = 0.2
    eval_epoch: int = 10
    save_epoch: int = 10
    conf_ratio: float = 1.0
    presaved_model_path: Path | str = None
    add_prompts: bool = False
    include_linear: bool = False

    assert aug in (
        'dropN',
        'permE',
        'maskN',
        'subgraph',
        'random',
        'random_v2',
        'dropout',
        'dropE',
        'featM'
        'none',
    )
    assert 0 <= aug_ratio <= 1
    assert 0 <= conf_ratio <= 1


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


class PretrainingLoader(Pretraining):
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    worker_init_fn: Callable = lambda x: _worker_seed_init(x, seed)


class MAELoader(_Config):
    mask_rate: float = 0.35


class TuningLoader(_Config):
    num_workers: int = 2
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


class GNN(_Config):
    num_layer: int = 5
    emb_dim: int = 300
    jk: str = "last"
    drop_ratio: float = 0

    # validity check
    assert jk in ("concat", "last", "max", "sum")
    assert 0 <= drop_ratio <= 1


class Encoder(_Config):
    num_layer: int = 5
    emb_dim: int = 300
    jk: str = "last"
    drop_ratio: int | float = 0

    # validity check
    assert jk in ("concat", "last", "max", "sum")
    assert 0 <= drop_ratio <= 1


class Decoder(_Config):
    hidden_dim: int = 300
    out_dim: int = 119


class GraphTrans(_Config):
    gnn_drop_ratio: float = 0
    d_model: int = 128
    trans_drop_ratio: float = 0.3


class BarlowTwins(_Config):
    lambda_: float = 0.0051
    sizes: tuple[int] = (300, 1200, 1200, 1200)


class Prompt(_Config):
    mode: str = 'add'
    assert mode in (
        'add',
        'mul',
        'mul_add',
    )


class Logger(_Config):
    level: int | str = logging.INFO


class OneSampleBN(_Config):
    is_enabled: bool = False
    strength: int = 16


class SSF(_Config):
    is_enabled: bool = False


class OnlineLearning(_Config):
    is_enabled: bool = False
    enable_tent_bn: bool = False
    optimize_tent_bn: bool = False
    use_shuffle: bool = False


_all_items = vars().values()


def get_all_configs() -> list[ConfigType]:
    results = []
    for i in _all_items:
        try:
            if issubclass(i, _Config) and not i.__name__.startswith('_'):
                results.append(i)
        except TypeError:
            pass
    return results
