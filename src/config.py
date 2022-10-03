import logging
import sys
from pathlib import Path
from typing import Callable
from importlib import reload

from src._config import Config
from src import utils

config_name: str | None = None
device: str = 'cuda:0'
seed: int = 0


class Paths(Config):
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


class Training(Config):
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 2048


class PretrainingDataset(Config):
    root: Path = Paths.datasets / 'zinc_standard_agent'
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


class PretrainingDataLoader(Config):

    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    worker_init_fn: Callable = lambda x: utils.worker_seed_init(x, seed)


class EqvGNN(Config):
    num_layer: int = 5
    emb_dim: int = 300
    use_edge_dropout: bool = False
    jk: str = "last"
    drop_ratio: int | float = 0

    # validity check
    assert jk in ("concat", "last", "max", "sum")
    assert 0 <= drop_ratio <= 1


class BarlowTwins(Config):
    lambda_: float = 0.0051
    sizes: tuple[int] = (300, 1200, 1200, 1200)


class Logger(Config):
    message_fmt: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    level: int | str = logging.INFO
    date_fmt: str = '%Y-%m-%d %H:%M:%S'

