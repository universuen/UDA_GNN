import pickle
from pathlib import Path

from torch import Tensor

import config


class History:
    def __init__(self, name: str = None):
        self.name = name
        self.values = []

    @property
    def avg_value(self):
        return sum(self.values) / len(self.values)

    def append(self, value: float | Tensor):
        if type(value) is Tensor:
            value = value.item()
        self.values.append(value)

    def save(self):
        path = config.Paths.results / config.config_name
        path.mkdir(exist_ok=True)
        with open(path / f'{self.name}.history', 'wb') as f:
            pickle.dump(
                {
                    'name': self.name,
                    'values': self.values,
                },
                f,
            )

    def load(self, path: Path):
        with open(path, 'rb') as f:
            records = pickle.load(f)
            self.name = records['name']
            self.values = records['values']

    def __repr__(self):
        return f'<History | {self.name} | {len(self.values)} values | {self.avg_value}>'
