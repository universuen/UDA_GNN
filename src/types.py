from torch.nn import Module

from src import logger


class GNNModel(Module):
    def __init__(self):
        super().__init__()
        self.logger = logger.Logger(self.__class__.__name__)


class PretrainingModel(Module):
    def __init__(self):
        super().__init__()
        self.logger = logger.Logger(self.__class__.__name__)


class Dataset():
    pass
