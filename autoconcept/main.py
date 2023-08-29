import os
import traceback

import hydra
import pytorch_lightning as pl
from callbacks import FreezingCallback
from clearml import Task
from extract import (compute_completeness, compute_disentanglement,
                     compute_informativeness, fit_linear_model,
                     prepare_data_dci, trace_interpretations)
from helpers import load_experiment, pretty_cfg, report_to_telegram, set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (DeviceStatsMonitor,
                                         LearningRateMonitor, ModelCheckpoint)

# import nltk

# nltk.download("wordnet")


def run(cfg):
    task = Task.init(project_name='alphacaption',
                     task_name='debug' if cfg.debug else cfg.name)
    task.upload_artifact(
        'outputs', artifact_object=os.path.abspath(os.getcwd()))
    set_seed(cfg.seed)

    print(1)

    pl.seed_everything(cfg.seed, workers=True)

    print(cfg.dataset)

    dm = instantiate(cfg.dataset)

    print(3)

    dm.setup()

    print(2)

    train_loader, test_loader, val_loader = dm.train_dataloader(
    ), dm.test_dataloader(), dm.val_dataloader()

    print(3)

    model = instantiate(cfg.model)

    print(4)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val",
        mode="min",
        filename="epoch{epoch:03d}-val_loss{loss/val:.5f}",
        auto_insert_metric_name=False,
        save_last=True,
    )

    print(5)

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

    # return f1_test


@hydra.main(version_base=None, config_path="config/conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg.name)
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
