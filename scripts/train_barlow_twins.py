import context

import torch
from torch_geometric.loader import DataLoader

import src
from src import config

if __name__ == '__main__':
    config.config_name = 'train_barlow_twins'
    # set logger
    logger = src.Logger('main')
    logger.log_config_info(config.Training)
    # set data loader
    loader = DataLoader(
        dataset=src.MoleculeDataset(),
        batch_size=config.PretrainingDataLoader.batch_size,
        shuffle=config.PretrainingDataLoader.shuffle,
        num_workers=config.PretrainingDataLoader.num_workers,
        pin_memory=config.PretrainingDataLoader.pin_memory,
        drop_last=config.PretrainingDataLoader.drop_last,
        worker_init_fn=config.PretrainingDataLoader.worker_init_fn,
    )
    model = src.model.pretraining.BarlowTwins().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), config.Training.lr)
    loss_history = src.LossHistory()
    for e in range(config.Training.epochs):
        for idx, (b1, b2, _) in enumerate(loader):
            torch.cuda.empty_cache()
            b1.to(config.device)
            b2.to(config.device)
            optimizer.zero_grad()
            loss = model(b1, b2)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)
            logger.info(src.utils.training_bar(e, idx, len(loader), loss=loss))
        if (e + 1) % 20 == 0:
            models_dir = config.Paths.models / config.config_name
            models_dir.mkdir(exist_ok=True)
            torch.save(
                model.state_dict(),
                models_dir / f'model_{e + 1}.pth'
            )
