import sys

import src
from src import config

if __name__ == '__main__':
    results_dir = sys.argv[1]

    results = {
        k: [] for k in config.datasets
    }

    for ds in results.keys():
        for seed in range(10):
            try:
                history = src.History()
                history.load(f'{results_dir}/{ds}_te_auc_{seed}.history')
                results[ds].append(history.avg_value)
            except FileNotFoundError:
                pass

    def safe_mean(list_):
        return 0 if len(list_) == 0 else sum(list_) / len(list_)

    values = list(map(safe_mean, results.values()))
    values.append(sum(values) / len(values))
    print(' '.join([*[f'{i: <10}' for i in results.keys()], 'mean']))
    print(' '.join([f'{i: <10.4f}' for i in values]))
