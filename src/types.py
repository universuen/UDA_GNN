from __future__ import annotations

from typing import Union

from torch.nn import Module

from src.logger import Logger
from src.history import History

Numeric = Union[int, float]


class GNNModel(Module):
    def __init__(self):
        super().__init__()
        self.logger = Logger(self.__class__.__name__)


class PretrainingModel(Module):
    def __init__(self):
        super().__init__()
        self.logger = Logger(self.__class__.__name__)


class Dataset:
    pass
