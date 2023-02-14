from pytorch_lightning.callbacks import Callback


class FreezingCallback(Callback):

    def __init__(self, epoch_freeze_backbone=50):
        super().__init__()
        if not epoch_freeze_backbone:
            epoch_freeze_backbone = -1
        self.epoch_freeze_backbone = epoch_freeze_backbone

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module.main, 'feature_extractor'):
            if trainer.current_epoch == self.epoch_freeze_backbone:
                for param in pl_module.main.feature_extractor.parameters():
                    param.requires_grad = False
                print(
                    f"Backbone's weights were frozen on {trainer.current_epoch} epoch!")
