import context

import sys

import src
import config

if __name__ == '__main__':
    config.config_name = sys.argv[1]

    results = {
        k: [] for k in config.datasets
    }

    for ds in results.keys():
        for seed in range(10):
            history = src.History()
            history.load(config.Paths.results / config.config_name / f'{ds}_tuning_losses_{seed}.history')
            results[ds].append(history.avg_value)

    print(' '.join(results.keys()))
    print(' '.join(map(lambda x: str(sum(x) / len(x)), results.values())))
