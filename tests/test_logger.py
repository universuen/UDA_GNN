from src.logger import Logger
from src import config, api

if __name__ == '__main__':
    config.config_name = 'test'
    logger = Logger('test')
    logger.info('You should see me in both console and log file.')
    api.log_all_config(logger)
