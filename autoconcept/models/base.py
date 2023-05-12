from functools import partial

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
from functional.gumbel import GumbelSigmoid
from models.feature_extractors.torchvision import TorchvisionFeatureExtractor
from models.helpers import AllMulticlassClfMetrics, retrieve
from models.predictors.mlp import MLPPredictor


class BaseModel(nn.Module):

    def __init__(self, extractor, predictor, interim_activation):
        super().__init__()
        assert extractor.out_features == predictor.layers[0]

        self.feature_extractor = extractor
        self.predictor = predictor
        self.interim_activation = interim_activation

        self.has_gumbel_sigmoid = isinstance(interim_activation, GumbelSigmoid)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(extractor.out_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, iteration):
        concept_logits = self.feature_extractor(x)
        concept_probs = self.sigmoid(concept_logits)

        args = [concept_logits]
        if self.has_gumbel_sigmoid:
            args.append(iteration)

        concept_activated = concept_logits
        if self.interim_activation:
            concept_activated = self.interim_activation(*args)

        concept_activated_bn = self.bn(concept_activated)
        prediction = self.predictor(concept_activated_bn)

        out_dict = dict(
            concept_logits=concept_logits,
            concept_probs=concept_probs,
            concept_activated=concept_activated,
            prediction=prediction,
        )

        return out_dict

    def inference(self, x, iteration=None):
        return self.softmax(self.forward(x, iteration)["prediction"])


class LitBaseModel(pl.LightningModule):

    def __init__(
        self,
        main=BaseModel(
            extractor=TorchvisionFeatureExtractor(),
            predictor=MLPPredictor(),
            interim_activation=nn.ReLU(),
        ),
        criterion=nn.CrossEntropyLoss(),
        optimizer_template=partial(torch.optim.Adam, lr=0.0001),
        scheduler_template=partial(
            torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.1),
        field="image",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['main'], logger=False)

        self.main = main
        self.criterion = criterion
        self.optimizer_template = optimizer_template
        self.scheduler_template = scheduler_template
        self.field = field

        # print("Predictor: ", self.main.predictor.main[0].weight)

    def forward(self, x, iteration=None):
        out_dict = self.main(x, iteration=iteration)
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

        if self.main.has_gumbel_sigmoid:
            self.log("gumbel/temp", self.main.interim_activation.t)

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
    base_model = LitBaseModel()
    x = torch.ones(2, 3, 299, 299)
    y = base_model(x)
    print(y)
