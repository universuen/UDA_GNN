import context

import config
import src

if __name__ == '__main__':
    config.config_name = 'test'
    history = src.History(values=[1, 2, 3])
    print(history.avg_value)
    print(history.max_value)
    print(history.std_deviation)
