import copy
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

    def __init__(self, feature_extractor, concept_extractor, predictor, interim_activation, predictor_aux=None):
        super().__init__()
        # assert feature_extractor.out_features == predictor.layers[0]
        # assert concept_extractor.out_features == predictor.layers[0]

        self.feature_extractor = feature_extractor
        self.concept_extractor = concept_extractor
        self.predictor = predictor
        self.predictor_aux = predictor_aux
        self.interim_activation = interim_activation
        self.interim_activation_aux = copy.deepcopy(interim_activation)

        self.has_gumbel_sigmoid = isinstance(interim_activation, GumbelSigmoid)
        self.sigmoid = nn.Sigmoid()
        self.bn_visual = nn.BatchNorm1d(feature_extractor.out_features)
        self.bn_textual = nn.BatchNorm1d(feature_extractor.out_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images, captions, iteration):
        feature_logits = self.feature_extractor(images)
        print(feature_logits[:, 0].min(), feature_logits[:, 0].max())

        concept_extractor_dict = self.concept_extractor(captions)
        concept_logits = concept_extractor_dict["concept_logits"]

        # print((feature_logits.min(), concept_logits.min()),
        #       (feature_logits.max(), concept_logits.max()),
        #       (feature_logits.abs().min(), concept_logits.abs().min())
        #       )

        # print("Features", torch.topk(
        #     feature_logits.flatten().abs(), 10, largest=False))
        # print("Concepts", torch.topk(
        #     concept_logits.flatten().abs(), 10, largest=False))

        feature_probs = self.sigmoid(feature_logits)
        concept_probs = self.sigmoid(concept_logits)

        args = [feature_logits]
        args_aux = [concept_logits]
        if self.has_gumbel_sigmoid:
            args.append(iteration)
            args_aux.append(iteration)

        feature_activated = feature_logits
        concept_activated = concept_logits
        if self.interim_activation:
            feature_activated = self.interim_activation(*args)
            concept_activated = self.interim_activation_aux(*args_aux)

        feature_activated_bn = self.bn_visual(feature_activated)
        concept_activated_bn = self.bn_textual(concept_activated)

        prediction = self.predictor(feature_activated_bn)

        prediction_aux = None
        if self.predictor_aux:
            prediction_aux = self.predictor_aux(concept_activated_bn)

        out_dict = dict(
            feature_logits=feature_logits,
            concept_logits=concept_logits,
            feature_probs=feature_probs,
            concept_probs=concept_probs,
            feature_activated=feature_activated,
            concept_activated=concept_activated,
            prediction=prediction,
            scores=concept_extractor_dict["scores"],
            scores_aux=concept_extractor_dict["scores_aux"]
        )

        if self.predictor_aux:
            out_dict["prediction_aux"] = prediction_aux

        if self.concept_extractor.regularize_distance:
            out_dict["loss_dist"] = concept_extractor_dict["loss_dist"]

        return out_dict

    def inference(self, images, iteration=None):
        feature_logits = self.feature_extractor(images)
        feature_probs = self.sigmoid(feature_logits)

        args = [feature_logits]
        if self.has_gumbel_sigmoid:
            args.append(iteration)

        feature_activated = feature_logits
        if self.interim_activation:
            feature_activated = self.interim_activation(*args)

        feature_activated = self.bn_visual(feature_activated)
        prediction = self.predictor(feature_activated)

        return self.softmax(prediction), feature_probs, feature_logits

    def inference_textual(self, indices, iteration=None):
        concept_extractor_dict = self.concept_extractor(indices)
        concept_logits = concept_extractor_dict["concept_logits"]
        return concept_logits


class LitAutoConceptBottleneckModel(pl.LightningModule):

    def __init__(
        self,
        main=AutoConceptBottleneckModel(
            feature_extractor=TorchvisionFeatureExtractor(),
            concept_extractor=ConceptExtractorAttention(vocab_size=100),
            predictor=MLPPredictor(),
            interim_activation=nn.ReLU(),
            predictor_aux=None,
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
        tie_weight=10,
        dist_weight=0.1,
        mix_tie_epoch=50,
        pretrain_embeddings_epoch=50,
        tie_loss_wrt_concepts=True,
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
        self.tie_weight = tie_weight
        self.dist_weight = dist_weight
        self.mix_tie_epoch = mix_tie_epoch
        self.pretrain_embeddings_epoch = pretrain_embeddings_epoch
        self.tie_loss_wrt_concepts = tie_loss_wrt_concepts
        self.last_iteration = 0

        self.automatic_optimization = False
        # print("Predictor: ", self.main.predictor.main[0].weight)

    def forward(self, images, indices, iteration=None):
        out_dict = self.main(images, indices, iteration=iteration)
        return out_dict

    def configure_optimizers(self):
        optimizer_model = self.optimizer_model_template(
            [*self.main.feature_extractor.parameters(), *self.main.bn_visual.parameters(), *self.main.predictor.parameters()])

        # optimizer_model = self.optimizer_model_template(self.parameters())

        parameters_textual = [
            *self.main.concept_extractor.parameters(), *self.main.bn_textual.parameters()]
        if self.main.predictor_aux:
            parameters_textual += [*self.main.predictor_aux.parameters()]
        optimizer_concept_extractor = self.optimizer_concept_extractor_template(
            parameters_textual)

        scheduler_model = self.scheduler_model_template(optimizer_model)
        scheduler_concept_extractor = self.scheduler_concept_extractor_template(
            optimizer_concept_extractor)
        return [optimizer_model, optimizer_concept_extractor], [scheduler_model, scheduler_concept_extractor]
        # return [optimizer_model], [scheduler_model]

    def training_step(self, batch, batch_idx):
        images, indices, target = batch["image"], batch["indices"], batch["target"]

        iteration = self.trainer.global_step
        if self.pretrain_embeddings_epoch:
            if self.trainer.current_epoch // self.pretrain_embeddings_epoch == 0:
                self.last_iteration = iteration
            else:
                iteration = max(0, iteration - self.last_iteration)
        out_dict = self(images, indices, iteration=iteration)

        prediction, feature_logits, concept_logits = out_dict[
            "prediction"], out_dict["feature_logits"], out_dict["concept_logits"]

        prediction_aux = None
        if "prediction_aux" in out_dict:
            prediction_aux = out_dict["prediction_aux"]

        tie_weight, dist_weight = self.tie_weight, self.dist_weight
        if self.mix_tie_epoch and self.trainer.current_epoch // self.mix_tie_epoch == 0:
            tie_weight, dist_weight = 0, 0

        task_weight, task_aux_weight = 1.0, 0.0
        if not self.pretrain_embeddings_epoch:
            task_aux_weight = 1.0
        if self.pretrain_embeddings_epoch and self.trainer.current_epoch // self.pretrain_embeddings_epoch == 0:
            task_weight, task_aux_weight = 0, 1.0

        loss_task = task_weight * self.criterion_task(prediction, target)
        loss_task_aux = None
        if prediction_aux is not None:
            loss_task_aux = task_aux_weight * \
                self.criterion_task(prediction_aux, target)

        tie_criterion_args = [feature_logits, concept_logits]
        if not self.tie_loss_wrt_concepts:
            tie_criterion_args = tie_criterion_args[::-1]
        loss_tie = tie_weight * self.criterion_tie(*tie_criterion_args)

        loss = loss_task + loss_tie
        if loss_task_aux is not None:
            loss += loss_task_aux

        loss_dist = torch.tensor(0.)
        if self.main.concept_extractor.regularize_distance:
            loss_dist = dist_weight * out_dict["loss_dist"]
            loss += loss_dist

        opt_clf, opt_tie = self.optimizers()
        opt_clf.zero_grad()
        opt_tie.zero_grad()

        # torch.autograd.set_detect_anomaly(True)

        self.manual_backward(loss)
        opt_clf.step()
        opt_tie.step()

        _target = retrieve(target)
        _prediction = retrieve(prediction.argmax(dim=1))

        _prediction_aux = None
        if "prediction_aux" in out_dict:
            _prediction_aux = retrieve(prediction_aux.argmax(dim=1))

        metrics = {
            "loss/train_task": loss_task,
            "loss/train_tie": loss_tie,
            "loss/train_dist": loss_dist,
            "loss/train": loss,
            "train/loss": loss,
            "train/loss_task": loss_task,
            "train/loss_tie": loss_tie,
            "train/loss_dist": loss_dist
        }
        if loss_task_aux is not None:
            metrics["loss/train_task_aux"] = loss_task_aux
            metrics["train/loss_task_aux"] = loss_task_aux
        self.log_dict(metrics)

        if self.main.has_gumbel_sigmoid:
            self.log("gumbel/temp", self.main.interim_activation.t)

        iter_dict = dict(
            loss=loss,
            loss_task=loss_task,
            loss_task_aux=loss_task_aux,
            loss_tie=loss_tie,
            loss_dist=loss_dist,
            target=_target,
            prediction=_prediction,
            prediction_aux=_prediction_aux,
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

        all_metrics = metrics.copy()
        if self.main.predictor_aux:
            all_prediction_aux = np.concatenate(
                [i['prediction_aux'] for i in outputs])
            metrics_aux = AllMulticlassClfMetrics()(
                all_target, all_prediction_aux, f'{phase}_aux')
            all_metrics.update(metrics_aux)
        self.log_dict(all_metrics)

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
        if self.pretrain_embeddings_epoch:
            if self.trainer.current_epoch // self.pretrain_embeddings_epoch == 0:
                self.last_iteration = iteration
            else:
                iteration = max(0, iteration - self.last_iteration)
        out_dict = self(images, indices, iteration=iteration)

        prediction, feature_logits, concept_logits = out_dict[
            "prediction"], out_dict["feature_logits"], out_dict["concept_logits"]

        prediction_aux = None
        if "prediction_aux" in out_dict:
            prediction_aux = out_dict["prediction_aux"]

        tie_weight, dist_weight = self.tie_weight, self.dist_weight
        if self.mix_tie_epoch and self.trainer.current_epoch // self.mix_tie_epoch == 0:
            tie_weight, dist_weight = 0, 0

        task_weight, task_aux_weight = 1.0, 0.0
        if not self.pretrain_embeddings_epoch:
            task_aux_weight = 1.0
        if self.pretrain_embeddings_epoch and self.trainer.current_epoch // self.pretrain_embeddings_epoch == 0:
            task_weight, task_aux_weight = 0, 1.0
        loss_task = task_weight * self.criterion_task(prediction, target)

        loss_task_aux = None
        if prediction_aux is not None:
            loss_task_aux = task_aux_weight * \
                self.criterion_task(prediction_aux, target)

        tie_criterion_args = [feature_logits, concept_logits]
        if not self.tie_loss_wrt_concepts:
            tie_criterion_args = tie_criterion_args[::-1]
        loss_tie = tie_weight * self.criterion_tie(*tie_criterion_args)

        loss = loss_task + loss_tie
        if loss_task_aux is not None:
            loss += loss_task_aux

        loss_dist = torch.tensor(0.)
        if self.main.concept_extractor.regularize_distance:
            loss_dist = dist_weight * out_dict["loss_dist"]
            loss += loss_dist

        _target = retrieve(target)
        _prediction = retrieve(prediction.argmax(dim=1))

        _prediction_aux = None
        if "prediction_aux" in out_dict:
            _prediction_aux = retrieve(prediction_aux.argmax(dim=1))

        metrics = {
            f"loss/{phase}_task": loss_task,
            f"loss/{phase}_tie": loss_tie,
            f"loss/{phase}_dist": loss_dist,
            f"loss/{phase}": loss,
            f"{phase}/loss": loss,
            f"{phase}/loss_task": loss_task,
            f"{phase}/loss_tie": loss_tie,
            f"{phase}/loss_dist": loss_dist
        }
        if loss_task_aux is not None:
            metrics[f"loss/{phase}_task_aux"] = loss_task_aux
            metrics[f"{phase}/loss_task_aux"] = loss_task_aux

        self.log_dict(metrics)

        if loss_task_aux is not None:
            loss_task_aux = loss_task_aux.item()

        iter_dict = dict(
            loss=loss.item(),
            loss_task=loss_task.item(),
            loss_task_aux=loss_task_aux,
            loss_tie=loss_tie.item(),
            loss_dist=loss_dist.item(),
            target=_target,
            prediction=_prediction,
            prediction_aux=_prediction_aux,
        )

        return iter_dict


if __name__ == '__main__':
    auto_concept_bottleneck_model = LitAutoConceptBottleneckModel().cuda()
    x = torch.ones(2, 3, 299, 299).cuda()
    c = torch.ones(2, 10).long().cuda()
    y = auto_concept_bottleneck_model(x, c)
    print(y)
