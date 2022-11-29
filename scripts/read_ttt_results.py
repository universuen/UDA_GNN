from __future__ import annotations
import context

import sys

import src
from src import config

if __name__ == '__main__':
    config.config_name = sys.argv[1]
    src.api.analyze_ttt_results_by_ratio(item_name='te_ttt_auc')
