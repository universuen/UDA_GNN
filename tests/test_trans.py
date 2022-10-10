from __future__ import annotations
import src

if __name__ == '__main__':
    src.api.set_debug_mode()
    loader = src.api.get_configured_tuning_dataloader(
        src.api.get_configured_tuning_dataset()
    )
    x = next(iter(loader))
    gt = src.api.get_configured_graph_trans()
    print(gt(x).shape)
