from __future__ import annotations
import context

import sys

import numpy as np
import pandas as pd

import src
from src import config, api


def analyze_results_temp():
    # config.config_name = sys.argv[1]
    config.datasets = [
        "bbbp",
        "tox21",
        "toxcast",
        "sider",
        "clintox",
        "muv",
        "hiv",
        "bace",
    ]

    ratios = [0.9, 1.0]
    results = {
        k: {
            kk: []
            for kk in ratios
        }
        for k in [*config.datasets, 'mean']
    }

    for ds in config.datasets:
        for ratio in ratios:
            for seed in config.loop_seeds:
                try:
                    history = api.get_configured_history(f'{ds}_te_auc_{seed}')
                    history.load()
                    results[ds][ratio].append(history[int(len(history.values) * ratio) - 1][0] * 100)
                except (FileNotFoundError, IndexError):
                    pass
            mean = api.safe_mean(results[ds][ratio])
            std = round(float(np.std(results[ds][ratio])), 1)
            results[ds][ratio] = f'{mean}±{std}'
            results['mean'][ratio].append((mean, std))

    for ratio in ratios:
        means = [i for i, _ in results['mean'][ratio]]
        results['mean'][ratio] = f"{api.safe_mean(means)}"

    pd.options.display.max_columns = None
    results = pd.DataFrame.from_dict(results)
    print(results)
    results.to_excel(config.Paths.results / config.config_name / 'analyzed_results.xlsx')


if __name__ == '__main__':
    for mr in (0.25, 0.35, 0.45):
        for d, pe in enumerate((10, 20, 50, 100)):
            config.config_name = f'1101_search_uda_ttt_pretrain_mr{mr}_pe{pe}'
            analyze_results_temp()
            