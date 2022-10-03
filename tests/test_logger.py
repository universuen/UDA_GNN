import context

from src.logger import Logger
import config

if __name__ == '__main__':
    config.config_name = 'test'
    logger = Logger('test')
    logger.info('You should see me in both console and log file.')
    logger.log_all_config()
