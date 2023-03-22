import torch.nn as nn
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
                print(
                    f"Backbone's weights were frozen on {trainer.current_epoch} epoch!")


class ReinitializeBottleneckCallback(Callback):

    def __init__(self, epoch_reinitialize=50):
        super().__init__()
        self.epoch_reinitialize = epoch_reinitialize

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module.main, 'feature_extractor'):
            if trainer.current_epoch == self.epoch_reinitialize - 1:
                bottleneck = pl_module.main.feature_extractor.main.fc
                in_features, out_features = bottleneck.in_features, bottleneck.out_features
                pl_module.main.feature_extractor.main.fc = nn.Linear(
                    in_features=in_features, out_features=out_features)
                print(
                    f"Bottleneck was re-initialized on {trainer.current_epoch} epoch!")
