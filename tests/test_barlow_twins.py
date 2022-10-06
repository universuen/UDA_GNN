from torch_geometric.loader import DataLoader

from src import config, api

if __name__ == '__main__':
    config.config_name = 'test'
    dataset = api.get_configured_pretraining_dataset()
    loader = api.get_configured_pretraining_loader(dataset)
    gnn = api.get_configured_gnn()
    bt = api.get_configured_barlow_twins(gnn)
    b_1, b_2, _ = next(iter(loader))
    print(bt(b_1, b_2))


