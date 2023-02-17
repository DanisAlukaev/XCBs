import os
import traceback

import hydra
import pytorch_lightning as pl
from callbacks import FreezingCallback
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

    train_loader, test_loader, val_loader = dm.train_dataloader(
    ), dm.test_dataloader(), dm.val_dataloader()

    model = instantiate(cfg.model)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val",
        mode="min",
        filename="epoch{epoch:03d}-val_loss{loss/val:.5f}",
        auto_insert_metric_name=False,
        save_last=True,
    )

    trainer_callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
        DeviceStatsMonitor(),
        FreezingCallback(cfg.epoch_freeze_backbone),
    ]

    if cfg.early_stopper:
        trainer_callbacks += [instantiate(cfg.early_stopper)]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=trainer_callbacks,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        log_every_n_steps=cfg.log_every_n_steps,
    )
    trainer.fit(model, train_loader, val_loader)

    checkpoint_path = checkpoint_callback.best_model_path
    target_class = get_class(cfg.model._target_)
    main = instantiate(cfg.model.main)
    inference = target_class.load_from_checkpoint(checkpoint_path, main=main)
    f1_test = trainer.test(inference, test_loader)[
        0]['test/weighted_avg/f1-score']
    return f1_test


@hydra.main(version_base=None, config_path="config/conf", config_name="config")
def main(cfg: DictConfig):
    try:
        metric = run(cfg)
        message = f"✅ Successful run from {cfg.timestamp}!\n\n"
        message += f"f1-score test: {metric}\n\n"
        message += f"Configuration:\n{pretty_cfg(cfg)}"
    except Exception:
        message = f"🚫 Run from {cfg.timestamp} failed!\n\n"
        message += traceback.format_exc()
        print(traceback.format_exc())
    report_to_telegram(message)


if __name__ == "__main__":
    main()
