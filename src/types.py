from __future__ import annotations

from typing import Union

from torch.nn import Module

from src.logger import Logger
from src.history import History
from src.model.gnn import GNNModel
from src.model.pretraining import PretrainingModel
from src.dataset import Dataset

Numeric = Union[int, float]


