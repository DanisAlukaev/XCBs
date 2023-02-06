
from functools import partial

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.base import BaseModel
from models.feature_extractors.torchvision import TorchvisionFeatureExtractor
from models.helpers import (AllMulticlassClfMetrics, MultiLabelClfMetrics,
                            retrieve)
from models.predictors.mlp import MLPPredictor


class LitConceptBottleneckModel(pl.LightningModule):

    def __init__(
        self,
        main=BaseModel(
            feature_extractor=TorchvisionFeatureExtractor(),
            predictor=MLPPredictor(),
            interim_activation=nn.ReLU(),
        ),
        criterion_task=nn.CrossEntropyLoss(),
        criterion_concept=nn.BCELoss(),
        lambda_p=0.01,
        optimizer_template=partial(torch.optim.Adam, lr=0.0001),
        scheduler_template=partial(
            torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.1),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['main'], logger=False)

        self.main = main
        self.criterion_task = criterion_task
        self.criterion_concept = criterion_concept
        self.lambda_p = lambda_p
        self.optimizer_template = optimizer_template
        self.scheduler_template = scheduler_template

    def forward(self, images, iteration=None):
        out_dict = self.main(images, iteration=iteration)
        return out_dict

    def configure_optimizers(self):
        optimizer = self.optimizer_template(self.parameters())
        scheduler = self.scheduler_template(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, target, attributes = batch["image"], batch["target"], batch["attributes"]

        iteration = self.trainer.global_step
        out_dict = self(images, iteration=iteration)

        prediction = out_dict["prediction"]
        concept_probs = out_dict["concept_probs"]

        loss_task = self.criterion_task(prediction, target)
        loss_concept = self.criterion_concept(concept_probs, attributes)

        loss = loss_task + self.lambda_p * loss_concept

        _target = retrieve(target)
        _prediction = retrieve(prediction.argmax(dim=1))

        _attributes = retrieve(attributes, flatten=False)
        _concepts = retrieve((concept_probs > 0.5).float(), flatten=False)

        metrics = {
            "loss/train": loss,
            "train/loss": loss,
            "train/loss_task": loss_task,
            "train/loss_concept": loss_concept
        }
        self.log_dict(metrics)

        if self.main.has_gumbel_sigmoid:
            self.log("gumbel/temp", self.main.interim_activation.t)

        iter_dict = dict(
            loss=loss,
            loss_task=loss_task,
            loss_concept=loss_concept,
            target=_target,
            prediction=_prediction,
            concepts=_concepts,
            attributes=_attributes,
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

        all_concepts = np.concatenate([i['concepts'] for i in outputs])
        all_attributes = np.concatenate([i['attributes'] for i in outputs])

        metrics = AllMulticlassClfMetrics()(
            all_target, all_prediction, f'{phase}')
        metrics.update(MultiLabelClfMetrics()(all_attributes, all_concepts))
        self.log_dict(metrics)

        mem = {
            'memory/gpu': torch.cuda.memory_allocated() / 1e9,
            'memory/vms': psutil.Process().memory_info().vms / 1e9,
            'memory/rss': psutil.Process().memory_info().rss / 1e9,
        }
        self.log_dict(mem, prog_bar=False)

        return metrics

    def _validation_step(self, batch, batch_idx, phase='val'):
        images, target, attributes = batch["image"], batch["target"], batch["attributes"]

        iteration = self.trainer.global_step
        out_dict = self(images, iteration=iteration)

        prediction = out_dict["prediction"]
        concept_probs = out_dict["concept_probs"]

        loss_task = self.criterion_task(prediction, target)
        loss_concept = self.criterion_concept(concept_probs, attributes)

        loss = loss_task + self.lambda_p * loss_concept

        _target = retrieve(target)
        _prediction = retrieve(prediction.argmax(dim=1))

        _attributes = retrieve(attributes, flatten=False)
        _concepts = retrieve((concept_probs > 0.5).float(), flatten=False)

        metrics = {
            f"loss/{phase}": loss,
            f"{phase}/loss": loss,
            f"{phase}/loss_task": loss_task,
            f"{phase}/loss_concept": loss_concept
        }
        self.log_dict(metrics)

        if self.main.has_gumbel_sigmoid:
            self.log("gumbel/temp", self.main.interim_activation.t)

        iter_dict = dict(
            loss=loss,
            loss_task=loss_task,
            loss_concept=loss_concept,
            target=_target,
            prediction=_prediction,
            concepts=_concepts,
            attributes=_attributes,
        )

        return iter_dict


if __name__ == '__main__':
    base_model = LitConceptBottleneckModel()
    x = torch.ones(2, 3, 299, 299)
    y = base_model(x)
    print(y)
