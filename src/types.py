from torch.nn import Module

from src.logger import Logger
from src.history import History


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
