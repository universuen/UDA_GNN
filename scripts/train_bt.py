import context

import torch
from torch_geometric.loader import DataLoader

import src
import config

if __name__ == '__main__':
    """
    Pretraining
    """
    # set config name
    config.config_name = 'barlow_twins_baseline'
    # set logger
    logger = src.Logger('main')
    # set data loader
    loader = DataLoader(
        dataset=src.dataset.MoleculeAugDataset(
            dataset=config.PretrainingDataset.dataset,
            aug_1=config.PretrainingDataset.aug_1,
            aug_ratio_1=config.PretrainingDataset.aug_ratio_1,
            aug_2=config.PretrainingDataset.aug_2,
            aug_ratio_2=config.PretrainingDataset.aug_ratio_2,
            use_original=config.PretrainingDataset.use_original,
        ),
        batch_size=config.Pretraining.batch_size,
        shuffle=True,
        num_workers=config.PretrainingLoader.num_workers,
        pin_memory=config.PretrainingLoader.pin_memory,
        drop_last=config.PretrainingLoader.drop_last,
        worker_init_fn=config.PretrainingLoader.worker_init_fn,
    )
    model = src.model.pretraining.BarlowTwins().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), config.Pretraining.lr)
    logger.log_config_info(config.Pretraining)
    loss_history = src.History('pretraining_losses')
    for e in range(config.Pretraining.epochs):
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

        models_dir = config.Paths.models / config.config_name
        models_dir.mkdir(exist_ok=True)
        torch.save(
            model.state_dict(),
            models_dir / f'pretraining_model_{e + 1}.pt'
        )
        logger.info(f"model saved at {models_dir / f'pretraining_model_{e + 1}.pt'}")
    # save the final model
    torch.save(
        model.gnn.state_dict(),
        config.Paths.models / config.config_name / f'pretraining_model_final.pt'
    )
    loss_history.save()

    """
    Tuning
    """
    for seed in range(10):
        config.seed = seed
        for ds in config.datasets:
            config.TuningDataset.dataset = ds
            src.utils.tune(config.TuningDataset.dataset)
