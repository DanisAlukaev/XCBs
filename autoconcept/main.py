import os
import traceback

import hydra
import pytorch_lightning as pl
from clearml import Task
from helpers import pretty_cfg, report_to_telegram, set_seed
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (DeviceStatsMonitor,
                                         LearningRateMonitor, ModelCheckpoint)


def run(cfg):
    task = Task.init(project_name='alphacaption',
                     task_name='debug' if cfg.debug else cfg.name)
    task.upload_artifact(
        'outputs', artifact_object=os.path.abspath(os.getcwd()))
    set_seed(cfg.seed)

    dm = instantiate(cfg.dataset)
    dm.setup()

    print(1)

    train_loader, test_loader, val_loader = dm.train_dataloader(
    ), dm.test_dataloader(), dm.val_dataloader()

    print(len(train_loader.dataset) / 64, len(val_loader.dataset) /
          64, len(test_loader.dataset) / 64)

    model = instantiate(cfg.model)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val",
        mode="min",
        filename="epoch{epoch:03d}-val_loss{loss/val:.5f}",
        auto_insert_metric_name=False,
    )

    trainer_callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
        DeviceStatsMonitor(),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=trainer_callbacks,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_loader, val_loader)

    checkpoint_path = checkpoint_callback.best_model_path
    target_class = get_class(cfg.model._target_)
    inference = target_class.load_from_checkpoint(checkpoint_path)
    trainer.test(inference, test_loader)


@hydra.main(version_base=None, config_path="config/conf", config_name="config")
def main(cfg: DictConfig):
    try:
        run(cfg)
        message = f"✅ Successful run from {cfg.timestamp}!\n\n"
        message += f"Configuration:\n{pretty_cfg(cfg)}"
    except Exception:
        message = f"🚫 Run from {cfg.timestamp} failed!\n\n"
        message += traceback.format_exc()
        print(traceback.format_exc())
    report_to_telegram(message)


if __name__ == "__main__":
    main()
