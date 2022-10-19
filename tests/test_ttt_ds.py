from src import api


if __name__ == '__main__':
    te_ds = api.get_configured_tuning_dataset()
    ds = api.get_configured_ttt_dataset(te_ds)
    pass
