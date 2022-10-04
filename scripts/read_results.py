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
            try:
                history = src.History()
                history.load(config.Paths.results / config.config_name / f'{ds}_tuning_losses_{seed}.history')
                results[ds].append(history.avg_value)
            except FileNotFoundError:
                pass

    values = list(map(max, results.values()))
    values.append(str(sum(values) / len(values)))
    print(' '.join([*results.keys(), 'mean']))
    print(' '.join([str(i) for i in values]))
