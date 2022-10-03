import logging
import sys
import os
from pathlib import Path

import config


class Logger(logging.Logger):
    def __init__(
            self,
            name: str,
            level: int | str = config.Logger.level,
            logs_dir: Path | None = None,
    ) -> None:
        if logs_dir is None:
            if config.config_name is None:
                raise NotImplementedError('config_name is mandatory.')
            logs_dir = config.Paths.logs / config.config_name
        super().__init__(name, level=level)
        # set format
        formatter = logging.Formatter(
            fmt=config.Logger.message_fmt,
            datefmt=config.Logger.date_fmt,
        )
        # set console output
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(formatter)
        s_handler.setLevel(config.Logger.level)
        self.addHandler(s_handler)
        # set file output
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / f'{name}.log'
        if os.path.exists(log_file):
            self.warning(f'{log_file} already exists!')
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(formatter)
        f_handler.setLevel(config.Logger.level)
        self.addHandler(f_handler)

    def log_config_info(self, config_cls: config.ConfigType):
        self.info(f'{config_cls.__name__:*^100}')
        for k, v in config_cls.to_dict().items():
            self.info(f'{k}: {v}')
        self.info('*' * 100)
