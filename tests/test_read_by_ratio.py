import context

from src import config, api


if __name__ == '__main__':
    config.config_name = 'debug'
    api.analyze_results_by_ratio()
