import logging

import torch
from models.feature_extractors.torchvision import TorchvisionFeatureExtractor
from models.helpers import weights_init
from models.predictors.mlp import MLPPredictor
from pytorch_lightning.callbacks import Callback


class FreezingCallback(Callback):

    def __init__(self, epoch_freeze_backbone=50):
        super().__init__()
        if not epoch_freeze_backbone:
            epoch_freeze_backbone = -1
        self.epoch_freeze_backbone = epoch_freeze_backbone

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module.main, 'feature_extractor'):
            if trainer.current_epoch == self.epoch_freeze_backbone - 1:
                for name, param in pl_module.main.feature_extractor.main.named_parameters():
                    if name.split(".")[0] != "fc":
                        param.requires_grad = False
                logging.info(
                    f"Backbone's weights were frozen on {trainer.current_epoch} epoch!")


class ReinitializeBottleneckCallback(Callback):

    def __init__(self, epoch_reinitialize=50):
        super().__init__()
        self.epoch_reinitialize = epoch_reinitialize

    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(pl_module.main, 'feature_extractor') and isinstance(pl_module.main.feature_extractor, TorchvisionFeatureExtractor):
            if trainer.current_epoch == self.epoch_reinitialize:
                torch.nn.init.xavier_uniform_(
                    pl_module.main.feature_extractor.main.fc.weight)
                logging.info("Bottleneck was re-initialized.")


class InitializePredictorCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(pl_module.main, 'predictor') and isinstance(pl_module.main.predictor, MLPPredictor):
            if trainer.current_epoch == 0:
                for l in pl_module.main.predictor.main:
                    if isinstance(l, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(l.weight)
                logging.info("Predictor was re-initialized.")

        if hasattr(pl_module.main, 'predictor_aux') and isinstance(pl_module.main.predictor_aux, MLPPredictor):
            if trainer.current_epoch == 0:
                for l in pl_module.main.predictor_aux.main:
                    if isinstance(l, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(l.weight)
                logging.info("Predictor aux was re-initialized.")


class ReinitializeTextualMLP(Callback):

    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.epoch:
            if hasattr(pl_module.main, 'concept_extractor'):
                torch.nn.init.xavier_uniform_(
                    pl_module.main.concept_extractor.queries_w.weight)
                torch.nn.init.xavier_uniform_(
                    pl_module.main.concept_extractor.values_w.weight)
                torch.nn.init.xavier_uniform_(
                    pl_module.main.concept_extractor.keys_w.weight)

                for mlp in pl_module.main.concept_extractor.mlps:
                    for l in mlp.main:
                        if isinstance(l, torch.nn.Linear):
                            torch.nn.init.xavier_uniform_(l.weight)
                logging.info("Textual MLP were re-initialized.")


class InitializeInceptionCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(pl_module.main, 'feature_extractor'):
            if trainer.current_epoch == 0:
                pl_module.main.feature_extractor.main.apply(weights_init)
                logging.info("Inception was re-initialized.")
