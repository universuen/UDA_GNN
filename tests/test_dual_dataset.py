import context

from src import config, api

if __name__ == '__main__':
    api.set_debug_mode()
    dual_ds = api.get_configured_dual_dataset()
    loader = api.get_configured_pretraining_loader(dual_ds)
    for i in loader:
        print(i)
