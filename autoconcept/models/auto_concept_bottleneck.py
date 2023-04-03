from functools import partial

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
from functional.gumbel import GumbelSigmoid
from functional.loss import KullbackLeiblerDivergenceLoss
from models.concept_extractors.transformer import ConceptExtractorAttention
from models.feature_extractors.torchvision import TorchvisionFeatureExtractor
from models.helpers import AllMulticlassClfMetrics, retrieve
from models.predictors.mlp import MLPPredictor


class AutoConceptBottleneckModel(nn.Module):

    def __init__(self, feature_extractor, concept_extractor, predictor, interim_activation, temperature=1.):
        super().__init__()
        assert feature_extractor.out_features == predictor.layers[0]
        assert concept_extractor.out_features == predictor.layers[0]

        self.feature_extractor = feature_extractor
        self.concept_extractor = concept_extractor
        self.predictor = predictor
        self.interim_activation = interim_activation
        self.T = temperature

        self.has_gumbel_sigmoid = isinstance(interim_activation, GumbelSigmoid)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(feature_extractor.out_features)

    def forward(self, images, captions, iteration):
        feature_logits = self.feature_extractor(images)
        # print("-" * 100)
        # print("Features: ", feature_logits.min(), feature_logits.max())
        concept_logits = self.concept_extractor(captions)

        feature_probs = self.sigmoid(feature_logits / self.T)
        concept_probs = self.sigmoid(concept_logits / self.T)

        args = [feature_logits]
        if self.has_gumbel_sigmoid:
            args.append(iteration)

        feature_activated = feature_logits
        if self.interim_activation:
            feature_activated = self.interim_activation(*args)

        feature_activated = self.bn(feature_activated)
        prediction = self.predictor(feature_activated)

        out_dict = dict(
            feature_logits=feature_logits,
            concept_logits=concept_logits,
            feature_probs=feature_probs,
            concept_probs=concept_probs,
            feature_activated=feature_activated,
            prediction=prediction,
        )

        return out_dict


class LitAutoConceptBottleneckModel(pl.LightningModule):

    def __init__(
        self,
        main=AutoConceptBottleneckModel(
            feature_extractor=TorchvisionFeatureExtractor(),
            concept_extractor=ConceptExtractorAttention(vocab_size=100),
            predictor=MLPPredictor(),
            interim_activation=nn.ReLU(),
            temperature=1.,
        ),
        criterion_task=nn.CrossEntropyLoss(),
        criterion_tie=KullbackLeiblerDivergenceLoss(),
        optimizer_model_template=partial(torch.optim.Adam, lr=0.0001),
        optimizer_concept_extractor_template=partial(
            torch.optim.Adam, lr=0.0001),
        scheduler_model_template=partial(
            torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.1),
        scheduler_concept_extractor_template=partial(
            torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.1),
        lambda_p=10,
        period=50,
        direct_kl=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['main'], logger=False)

        self.main = main
        self.criterion_task = criterion_task
        self.criterion_tie = criterion_tie
        self.optimizer_model_template = optimizer_model_template
        self.optimizer_concept_extractor_template = optimizer_concept_extractor_template
        self.scheduler_model_template = scheduler_model_template
        self.scheduler_concept_extractor_template = scheduler_concept_extractor_template
        self.lambda_p = lambda_p
        self.period = period
        self.direct_kl = direct_kl

        self.automatic_optimization = False

    def forward(self, images, indices, iteration=None):
        out_dict = self.main(images, indices, iteration=iteration)
        return out_dict

    def configure_optimizers(self):
        optimizer_model = self.optimizer_model_template(
            [*self.main.feature_extractor.parameters(), *self.main.predictor.parameters()])
        optimizer_concept_extractor = self.optimizer_concept_extractor_template(
            self.main.concept_extractor.parameters())
        scheduler_model = self.scheduler_model_template(optimizer_model)
        scheduler_concept_extractor = self.scheduler_concept_extractor_template(
            optimizer_concept_extractor)
        return [optimizer_model, optimizer_concept_extractor], [scheduler_model, scheduler_concept_extractor]

    def training_step(self, batch, batch_idx):
        images, indices, target = batch["image"], batch["indices"], batch["target"]

        iteration = self.trainer.global_step
        out_dict = self(images, indices, iteration=iteration)

        prediction, feature_probs, concept_probs = out_dict[
            "prediction"], out_dict["feature_probs"], out_dict["concept_probs"]

        # TODO: lambda_p should be from 0 to 1
        lambda_p = self.lambda_p if self.trainer.current_epoch // self.period > 0 else 0

        loss_task = self.criterion_task(prediction, target)
        if self.direct_kl:
            loss_tie = lambda_p * \
                self.criterion_tie(concept_probs, feature_probs)
        else:
            loss_tie = lambda_p * \
                self.criterion_tie(feature_probs, concept_probs)

        # TODO: multiplier for tie loss
        loss = loss_task + loss_tie

        opt_clf, opt_tie = self.optimizers()

        opt_clf.zero_grad()
        opt_tie.zero_grad()

        self.manual_backward(loss)

        # for name, param in self.main.named_parameters():
        #     print(name, torch.isfinite(param.grad).all())

        # nn.utils.clip_grad_value_(self.main.parameters(), clip_value=0.01)

        opt_clf.step()
        opt_tie.step()

        torch.autograd.set_detect_anomaly(True)

        _target = retrieve(target)
        _prediction = retrieve(prediction.argmax(dim=1))

        metrics = {
            "loss/train_task": loss_task,
            "loss/train_tie": loss_tie,
            "loss/train": loss,
            "train/loss": loss,
            "train/loss_task": loss_task,
            "train/loss_tie": loss_tie
        }
        self.log_dict(metrics)

        if self.main.has_gumbel_sigmoid:
            self.log("gumbel/temp", self.main.interim_activation.t)

        iter_dict = dict(
            loss=loss,
            loss_task=loss_task,
            loss_tie=loss_tie,
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
        sch_clf, sch_tie = self.lr_schedulers()
        sch_clf.step()
        sch_tie.step()

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
        images, indices, target = batch["image"], batch["indices"], batch["target"]

        iteration = self.trainer.global_step
        out_dict = self(images, indices, iteration=iteration)

        prediction, feature_probs, concept_probs = out_dict[
            "prediction"], out_dict["feature_probs"], out_dict["concept_probs"]

        lambda_p = self.lambda_p if self.trainer.current_epoch // self.period > 0 else 0

        loss_task = self.criterion_task(prediction, target)
        if self.direct_kl:
            loss_tie = lambda_p * \
                self.criterion_tie(concept_probs, feature_probs)
        else:
            loss_tie = lambda_p * \
                self.criterion_tie(feature_probs, concept_probs)

        # TODO: multiplier for tie loss
        loss = loss_task + loss_tie

        _target = retrieve(target)
        _prediction = retrieve(prediction.argmax(dim=1))

        metrics = {
            f"loss/{phase}_task": loss_task,
            f"loss/{phase}_tie": loss_tie,
            f"loss/{phase}": loss,
            f"{phase}/loss": loss,
            f"{phase}/loss_task": loss_task,
            f"{phase}/loss_tie": loss_tie
        }
        self.log_dict(metrics)

        iter_dict = dict(
            loss=loss.item(),
            loss_task=loss_task.item(),
            loss_tie=loss_tie.item(),
            target=_target,
            prediction=_prediction
        )

        return iter_dict


if __name__ == '__main__':
    auto_concept_bottleneck_model = LitAutoConceptBottleneckModel().cuda()
    x = torch.ones(2, 3, 299, 299).cuda()
    c = torch.ones(2, 10).long().cuda()
    y = auto_concept_bottleneck_model(x, c)
    print(y)
