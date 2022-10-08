from __future__ import annotations
import context

import sys

import pandas as pd

import src
from src import config, types


def safe_mean(list_: list[types.Numeric]) -> types.Numeric:
    return 0 if len(list_) == 0 else round(sum(list_) / len(list_), 1)


if __name__ == '__main__':
    results_dir = sys.argv[1]
    steps = list(range(10, 101, 10))

    results = {
        k: {
            kk: []
            for kk in steps
        }
        for k in [*config.datasets, 'mean']
    }

    for ds in config.datasets:
        for step in steps:
            for seed in config.loop_seeds:
                try:
                    history = src.History()
                    history.load(f'{results_dir}/{ds}_te_auc_{seed}.history')
                    results[ds][step].append(history[step - 1] * 100)
                except (FileNotFoundError, IndexError):
                    pass
            results[ds][step] = safe_mean(results[ds][step])
            results['mean'][step].append(results[ds][step])

    for step in steps:
        results['mean'][step] = safe_mean(results['mean'][step])

    results = pd.DataFrame.from_dict(results)
    print(results)
    results.to_excel('results.xlsx')
