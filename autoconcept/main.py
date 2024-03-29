import logging
import os
import traceback

import hydra
import pytorch_lightning as pl
from callbacks import FreezingCallback
from clearml import Task
from extract import (compute_completeness, compute_disentanglement,
                     compute_informativeness, fit_linear_model,
                     prepare_data_dci, trace_interpretations)
from helpers import (get_scalar_page_clearml, load_experiment,
                     report_to_telegram, set_seed)
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (DeviceStatsMonitor,
                                         LearningRateMonitor, ModelCheckpoint)


def init_clearml_task(cfg):
    task = Task.init(project_name='alphacaption',
                     task_name='debug' if cfg.debug else cfg.name)
    task.upload_artifact(
        'outputs', artifact_object=os.path.abspath(os.getcwd()))
    return task


def run(task, cfg):

    set_seed(cfg.seed)

    pl.seed_everything(cfg.seed, workers=True)

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
        # InitializePredictorCallback(),
        # ReinitializeTextualMLP(0),
    ]
    # TODO: re-initialization worsens performance
    # if cfg.reinitialize_feature_extractor:
    #     trainer_callbacks.append(InitializeInceptionCallback())
    # if isinstance(cfg.epoch_reinitialize, int):
    #     trainer_callbacks += [ReinitializeBottleneckCallback(cfg.epoch_reinitialize)]

    if cfg.early_stopper:
        trainer_callbacks += [instantiate(cfg.early_stopper)]

    if isinstance(cfg.epoch_freeze_backbone, int):
        trainer_callbacks += [FreezingCallback(cfg.epoch_freeze_backbone)]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=trainer_callbacks,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        log_every_n_steps=cfg.log_every_n_steps,
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

    dm, inference = load_experiment(".")
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    X_train, y_train = prepare_data_dci(train_loader, inference.cuda())
    X_test, y_test = prepare_data_dci(test_loader, inference.cuda())

    R, errors = fit_linear_model(
        X_train, y_train, X_test, y_test, seed=cfg.seed, fast=True)
    disentanglement = compute_disentanglement(R)
    completeness = compute_completeness(R)
    informativeness = compute_informativeness(errors)

    logger = task.get_logger()
    logger.report_scalar("disentanglement", "test", disentanglement, 0)
    logger.report_scalar("completeness", "test", completeness, 0)
    logger.report_scalar("informativeness", "test", informativeness, 0)

    dm, inference = load_experiment(".")
    test_loader = dm.test_dataloader()

    f1_test = trainer.test(inference, test_loader)[
        0]['test/weighted_avg/f1-score']

    dm, inference = load_experiment(".")

    if cfg.trace_interpretations:
        trace_interpretations(dm, inference)

    return f1_test


@hydra.main(version_base=None, config_path="config/conf", config_name="config")
def main(cfg: DictConfig):
    task = init_clearml_task(cfg)
    logs_url = task.get_output_log_web_page()
    scalar_url = get_scalar_page_clearml(logs_url)
    try:
        metric = run(task, cfg)
        message = f"✅ Successful run from {cfg.timestamp}!\n\n"
        message += f"f1-score test: {metric}\n\n"
        message += f"ClearML: {scalar_url}"
    except Exception:
        message = f"🚫 Run from {cfg.timestamp} failed!\n\n"
        message += f"ClearML: {logs_url}\n\n"
        logging.error(traceback.format_exc())
    report_to_telegram(message)


if __name__ == "__main__":
    main()
