from functools import partial

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.helpers import AllMulticlassClfMetrics, retrieve
from models.predictors.mlp import MLPPredictor


class PredictorModel(nn.Module):

    def __init__(self, predictor):
        super().__init__()

        self.predictor = predictor
        self.bn = nn.BatchNorm1d(predictor.layers[0])

    def forward(self, x):
        prediction = self.predictor(self.bn(x))

        out_dict = dict(
            prediction=prediction,
        )

        return out_dict


class LitPredictorModel(pl.LightningModule):

    def __init__(
        self,
        main=PredictorModel(
            predictor=MLPPredictor(),
        ),
        criterion=nn.CrossEntropyLoss(),
        optimizer_template=partial(torch.optim.Adam, lr=0.0001),
        scheduler_template=partial(
            torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.1),
        field="attributes",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['main'], logger=False)

        self.main = main
        self.criterion = criterion
        self.optimizer_template = optimizer_template
        self.scheduler_template = scheduler_template
        self.field = field

    def forward(self, images, iteration=None):
        out_dict = self.main(images)
        return out_dict

    def configure_optimizers(self):
        optimizer = self.optimizer_template(self.parameters())
        scheduler = self.scheduler_template(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, target = batch[self.field], batch["target"]

        iteration = self.trainer.global_step
        out_dict = self(x, iteration=iteration)

        prediction = out_dict["prediction"]
        loss = self.criterion(prediction, target)

        _target = retrieve(target)
        _prediction = retrieve(prediction.argmax(dim=1))

        metrics = {
            "loss/train": loss,
            "train/loss": loss
        }
        self.log_dict(metrics)

        iter_dict = dict(
            loss=loss,
            target=_target,
            prediction=_prediction
        )

        return iter_dict

    def validation_step(self, batch, batch_idx):
        metrics = self._validation_step(batch, batch_idx, phase='val')
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self._validation_step(batch, batch_idx, phase='test')
        return metrics

    def training_epoch_end(self, outputs):
        self._validation_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        return self._validation_epoch_end(outputs, phase='val')

    def test_epoch_end(self, outputs):
        return self._validation_epoch_end(outputs, phase='test')

    def _validation_epoch_end(self, outputs, phase='val'):
        all_prediction = np.concatenate([i['prediction'] for i in outputs])
        all_target = np.concatenate([i['target'] for i in outputs])

        metrics = AllMulticlassClfMetrics()(
            all_target, all_prediction, f'{phase}')
        self.log_dict(metrics)

        mem = {
            'memory/gpu': torch.cuda.memory_allocated() / 1e9,
            'memory/vms': psutil.Process().memory_info().vms / 1e9,
            'memory/rss': psutil.Process().memory_info().rss / 1e9,
        }
        self.log_dict(mem, prog_bar=False)

        return metrics

    def _validation_step(self, batch, batch_idx, phase='val'):
        x, target = batch[self.field], batch["target"]

        iteration = self.trainer.global_step
        out_dict = self(x, iteration=iteration)

        prediction = out_dict["prediction"]
        loss = self.criterion(prediction, target)

        _target = retrieve(target)
        _prediction = retrieve(prediction.argmax(dim=1))

        metrics = {
            f"loss/{phase}": loss,
            f"{phase}/loss": loss
        }
        self.log_dict(metrics)

        iter_dict = dict(
            loss=loss.item(),
            target=_target,
            prediction=_prediction
        )

        return iter_dict


if __name__ == '__main__':
    base_model = LitPredictorModel()
    x = torch.ones(2, 3, 299, 299)
    y = base_model(x)
    print(y)
