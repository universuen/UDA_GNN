from __future__ import annotations
import context

import sys

import src
from src import config

if __name__ == '__main__':
    # config.config_name = sys.argv[1]
    config.config_name = 'finetune_baseline'
    src.api.analyze_results_by_ratio()
