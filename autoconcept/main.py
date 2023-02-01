import logging
import os
import traceback

import hydra
import psutil
from clearml import Task
from helpers import pretty_cfg, report_to_telegram, set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig


def run(cfg):
    task = Task.init(project_name='alphacaption',
                     task_name='debug' if cfg.debug else cfg.name)
    task.upload_artifact(
        'outputs', artifact_object=os.path.abspath(os.getcwd()))
    set_seed(cfg.seed)

    logging.info(
        f'RAM/train/rss {psutil.Process().memory_info().rss / 1e9} RAM/train/vms {psutil.Process().memory_info().vms / 1e9}')

    dm = instantiate(cfg.dataset)
    dm.setup()

    train_loader, test_loader, val_loader = dm.train_dataloader(
    ), dm.test_dataloader(), dm.val_dataloader()

    print(len(train_loader.dataset))


@hydra.main(version_base=None, config_path="config/conf", config_name="config")
def main(cfg: DictConfig):
    try:
        run(cfg)
        message = f"✅ Successful run from {cfg.timestamp}!\n\n"
        message += f"Configuration:\n{pretty_cfg(cfg)}"
    except Exception:
        message = f"🚫 Run from {cfg.timestamp} failed!\n\n"
        message += traceback.format_exc()
    report_to_telegram(message)


if __name__ == "__main__":
    main()
