import json
import logging
import math

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


def prepare_data_dci(loader, model):
    X, y = list(), list()
    is_framework = hasattr(model.main, "concept_extractor")

    for batch in tqdm(loader):
        images, attributes_all = batch["image"].cuda(), batch["attributes"]
        N = images.shape[0]

        if is_framework:
            batch_features = model.main.inference(
                images)[1].cpu().detach().numpy()
        else:
            batch_features = model(
                images)["concept_probs"].cpu().detach().numpy()

        for sample_id in range(N):
            attributes = np.array(attributes_all[sample_id])
            features = batch_features[sample_id]

            X.append(features)
            y.append(attributes)

    X, y = np.array(X), np.array(y)

    return X, y


TINY = 1e-12


def norm_entropy(p):
    n = p.shape[0]
    return - p.dot(np.log(p + TINY) / np.log(n + TINY))


def entropic_scores(r):
    r = np.abs(r)
    ps = r / np.sum(r, axis=0)  # 'probabilities'
    hs = [1-norm_entropy(p) for p in ps.T]
    return hs


def mse(predicted, target):
    predicted = predicted[:, None] if len(
        predicted.shape) == 1 else predicted  # (n,)->(n,1)
    target = target[:, None] if len(
        target.shape) == 1 else target  # (n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0]  # value not array


def rmse(predicted, target):
    return np.sqrt(mse(predicted, target))


def nrmse(predicted, target):
    return rmse(predicted, target) / np.std(target)


def fit_linear_model(X_train, y_train, X_test, y_test, err_fn=nrmse, fast=False, seed=42):
    n_attributes = y_train.shape[1]
    R, errors = list(), list()
    for regressor_idx in tqdm(range(n_attributes)):
        kwargs = {"random_state": seed}
        if fast:
            kwargs["n_estimators"] = 20
            kwargs["max_depth"] = 10
        regressor = RandomForestRegressor(**kwargs)
        regressor.fit(X_train, y_train[:, regressor_idx])
        y_pred = regressor.predict(X_test)
        errors.append(err_fn(y_pred, y_test[:, regressor_idx]))
        R.append(regressor.feature_importances_)
    return np.array(R), np.array(errors)


def compute_disentanglement(R):
    disent_scores = entropic_scores(R.T)
    c_rel_importance = np.sum(R, 1) / np.sum(R)
    disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)
    return disent_w_avg


def compute_completeness(R):
    complete_scores = entropic_scores(R)
    complete_scores = [v for v in complete_scores if not math.isnan(v)]
    complete_avg = np.mean(complete_scores)
    return complete_avg


def compute_informativeness(errors):
    informativeness = np.mean(errors)
    return informativeness


def trace_interpretations(dm, model):
    model = model.cuda()
    train_loader = dm.train_dataloader()

    vocab_size = len(dm.dataloader_kwargs['collate_fn'].vocabulary.vocab)
    top_k_tokens = vocab_size

    n_concepts = model.main.concept_extractor.queries_w.num_embeddings
    logging.info(f"Vocab size: {vocab_size}")

    results = [dict(concept=None, feature=None) for _ in range(n_concepts)]

    # textual part
    distributions = [np.zeros(vocab_size) for _ in range(n_concepts)]
    dummy_distributions = [0. for _ in range(n_concepts)]
    n_tokens = np.zeros(vocab_size)

    itos_map = dm.dataloader_kwargs['collate_fn'].vocabulary.vocab.get_itos()

    logging.info("Processing train data via concept extractor...")
    for batch in tqdm(train_loader):
        input_ids = batch["indices"].cuda()
        N, _ = input_ids.shape
        out_dict = model.main.concept_extractor(input_ids)
        scores = out_dict["scores"]
        scores_aux = out_dict["scores_aux"]
        no_dummy_tokens = scores.shape == scores_aux.shape

        for concept_idx in range(n_concepts):
            concept_score = scores[:, concept_idx, :]
            concept_score_aux = scores_aux[:, concept_idx, -1]
            concept_score = concept_score.squeeze().cpu().detach().numpy()
            concept_score_aux = concept_score_aux.squeeze().cpu().detach().numpy()

            for sample_idx in range(N):
                sample_token_ids = input_ids[sample_idx]
                sample_scores = concept_score[sample_idx]

                sample_score_aux = concept_score_aux[sample_idx]
                dummy_distributions[concept_idx] += sample_score_aux

                for token_idx, token_id in enumerate(sample_token_ids):
                    distributions[concept_idx][token_id] += sample_scores[token_idx]

                    if concept_idx == n_concepts - 1:
                        n_tokens[token_id] += 1

    dummy_distributions = np.array(
        dummy_distributions) / len(train_loader.dataset)
    distributions = np.nan_to_num(np.array(distributions) / n_tokens)

    logging.info("Export results...")
    for concept_idx in tqdm(range(n_concepts)):
        concept_distribution = distributions[concept_idx]
        token_ids_sorted = (-concept_distribution).argsort()[:top_k_tokens]

        scores_per_token = list()
        for token_id in token_ids_sorted:
            token_score = concept_distribution[token_id]
            token = itos_map[token_id]
            scores_per_token.append([token, token_score])

        if not no_dummy_tokens:
            scores_per_token.append(
                [f"<d{concept_idx}>", dummy_distributions[concept_idx]])

        scores_per_token = sorted(scores_per_token, key=lambda x: x[1])
        scores_per_token = sorted(
            scores_per_token, key=lambda x: x[1], reverse=True)
        results[concept_idx]["concept"] = scores_per_token

    # visual part
    n_features = model.main.feature_extractor.main.fc.out_features
    top_k = 10

    logging.info("Processing train data via feature extractor...")
    per_feature_logits = dict(
        lrg=[list() for _ in range(n_features)],
        mdm=[list() for _ in range(n_features)],
        sml=[list() for _ in range(n_features)]
    )
    for batch in tqdm(train_loader):
        images = batch["image"].cuda()
        filenames = batch["img_path"]

        logits = model.main.feature_extractor(images)
        logits = logits.cpu().detach()

        lrg_topk = torch.topk(logits.T, k=top_k, dim=1, largest=True)
        lrg_val, lrg_idx = lrg_topk.values, lrg_topk.indices
        lrg_filenames = np.array([filenames[i]
                                 for i in lrg_idx.flatten().tolist()])
        lrg_filenames = lrg_filenames.reshape(lrg_idx.shape)
        lrg_pairs_ = [list(zip(lrg_filenames[i], lrg_val[i]))
                      for i in range(n_features)]
        lrg_pairs = list()
        for lrg_pair in lrg_pairs_:
            lrg_pairs.append(list())
            for (a, b) in lrg_pair:
                lrg_pairs[-1].append((a, b.item()))
        per_feature_logits["lrg"] = [sorted(
            a + b, key=lambda x: x[1]) for a, b in zip(per_feature_logits["lrg"], lrg_pairs)]
        per_feature_logits["lrg"] = [sorted(a, reverse=True, key=lambda x: x[1])[
            :top_k] for a in per_feature_logits["lrg"]]

        mdm_topk = torch.topk(logits.abs().T, k=top_k, dim=1, largest=False)
        mdm_val, mdm_idx = mdm_topk.values, mdm_topk.indices
        mdm_filenames = np.array([filenames[i]
                                 for i in mdm_idx.flatten().tolist()])
        mdm_filenames = mdm_filenames.reshape(mdm_idx.shape)
        mdm_pairs_ = [list(zip(mdm_filenames[i], mdm_val[i]))
                      for i in range(n_features)]
        mdm_pairs = list()
        for mdm_pair in mdm_pairs_:
            mdm_pairs.append(list())
            for (a, b) in mdm_pair:
                mdm_pairs[-1].append((a, b.item()))
        per_feature_logits["mdm"] = [sorted(
            a + b, key=lambda x: x[1]) for a, b in zip(per_feature_logits["mdm"], mdm_pairs)]
        per_feature_logits["mdm"] = [sorted(a, reverse=False, key=lambda x: abs(x[1]))[
            :top_k] for a in per_feature_logits["mdm"]]

        sml_topk = torch.topk(logits.T, k=top_k, dim=1, largest=False)
        sml_val, sml_idx = sml_topk.values, sml_topk.indices
        sml_filenames = np.array([filenames[i]
                                 for i in sml_idx.flatten().tolist()])
        sml_filenames = sml_filenames.reshape(sml_idx.shape)
        sml_pairs_ = [list(zip(sml_filenames[i], sml_val[i]))
                      for i in range(n_features)]
        sml_pairs = list()
        for sml_pair in sml_pairs_:
            sml_pairs.append(list())
            for (a, b) in sml_pair:
                sml_pairs[-1].append((a, b.item()))
        per_feature_logits["sml"] = [sorted(
            a + b, key=lambda x: x[1]) for a, b in zip(per_feature_logits["sml"], sml_pairs)]
        per_feature_logits["sml"] = [sorted(a, reverse=False, key=lambda x: x[1])[
            :top_k] for a in per_feature_logits["sml"]]

    logging.info("Export results...")
    for feature_idx in tqdm(range(n_features)):
        if not results[feature_idx]["feature"]:
            results[feature_idx]["feature"] = dict()
        results[feature_idx]["feature"]["lrg"] = per_feature_logits["lrg"][feature_idx]
        results[feature_idx]["feature"]["mdm"] = per_feature_logits["mdm"][feature_idx]
        results[feature_idx]["feature"]["sml"] = per_feature_logits["sml"][feature_idx]

    results = {
        "results": results,
        "top_k": top_k
    }

    with open("./results.json", "w") as outfile:
        json.dump(results, outfile)

    return True
