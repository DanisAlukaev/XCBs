import json

import numpy as np
import torch
from tqdm import tqdm


def trace_interpretations(dm, model):
    print(next(model.parameters()).is_cuda)
    model = model.cuda()
    train_loader = dm.train_dataloader()

    vocab_size = len(dm.dataloader_kwargs['collate_fn'].vocabulary.vocab)
    top_k_tokens = vocab_size

    n_concepts = model.main.concept_extractor.queries_w.num_embeddings
    print(f"Vocab size: {vocab_size}")

    results = [dict(concept=None, feature=None) for _ in range(n_concepts)]

    # textual part
    distributions = [np.zeros(vocab_size) for _ in range(n_concepts)]
    n_tokens = np.zeros(vocab_size)

    itos_map = dm.dataloader_kwargs['collate_fn'].vocabulary.vocab.get_itos()

    print("Processing train data via concept extractor...")
    for batch in tqdm(train_loader):
        input_ids = batch["indices"].cuda()
        N, _ = input_ids.shape
        out_dict = model.main.concept_extractor(input_ids)
        scores = out_dict["scores"]

        for concept_idx in range(n_concepts):
            concept_score = scores[:, concept_idx, :]
            concept_score = concept_score.squeeze().cpu().detach().numpy()

            for sample_idx in range(N):
                sample_token_ids = input_ids[sample_idx]
                sample_scores = concept_score[sample_idx]

                for token_idx, token_id in enumerate(sample_token_ids):
                    distributions[concept_idx][token_id] += sample_scores[token_idx]

                    if concept_idx == n_concepts - 1:
                        n_tokens[token_id] += 1

    distributions = np.array(distributions) / n_tokens

    print("Export results...")
    for concept_idx in tqdm(range(n_concepts)):
        concept_distribution = distributions[concept_idx]
        token_ids_sorted = (-concept_distribution).argsort()[:top_k_tokens]

        scores_per_token = list()
        for token_id in token_ids_sorted:
            token_score = concept_distribution[token_id]
            token = itos_map[token_id]
            scores_per_token.append([token, token_score])

        results[concept_idx]["concept"] = scores_per_token

    # visual part
    n_features = model.main.feature_extractor.main.fc.out_features
    top_k = 10

    print("Processing train data via feature extractor...")
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
        lrg_pairs = [list(zip(lrg_filenames[i], lrg_val[i]))
                     for i in range(n_features)]
        per_feature_logits["lrg"] = [sorted(a + b, reverse=True, key=lambda x: x[1])[
            :top_k] for a, b in zip(per_feature_logits["lrg"], lrg_pairs)]

        mdm_topk = torch.topk(logits.abs().T, k=top_k, dim=1, largest=False)
        mdm_val, mdm_idx = mdm_topk.values, mdm_topk.indices
        mdm_filenames = np.array([filenames[i]
                                 for i in mdm_idx.flatten().tolist()])
        mdm_filenames = mdm_filenames.reshape(mdm_idx.shape)
        mdm_pairs = [list(zip(mdm_filenames[i], mdm_val[i]))
                     for i in range(n_features)]
        per_feature_logits["mdm"] = [sorted(a + b, reverse=False, key=lambda x: abs(
            x[1]))[:top_k] for a, b in zip(per_feature_logits["mdm"], mdm_pairs)]

        sml_topk = torch.topk(logits.T, k=top_k, dim=1, largest=False)
        sml_val, sml_idx = sml_topk.values, sml_topk.indices
        sml_filenames = np.array([filenames[i]
                                 for i in sml_idx.flatten().tolist()])
        sml_filenames = sml_filenames.reshape(sml_idx.shape)
        sml_pairs = [list(zip(sml_filenames[i], sml_val[i]))
                     for i in range(n_features)]
        per_feature_logits["sml"] = [sorted(a + b, reverse=False, key=lambda x: x[1])[
            :top_k] for a, b in zip(per_feature_logits["sml"], sml_pairs)]

    print("Export results...")
    for feature_idx in tqdm(range(n_features)):
        if not results[feature_idx]["feature"]:
            results[feature_idx]["feature"] = dict()
        results[feature_idx]["feature"]["lrg"] = [
            (a, b.item()) for a, b in per_feature_logits["lrg"][feature_idx]]
        results[feature_idx]["feature"]["mdm"] = [
            (a, b.item()) for a, b in per_feature_logits["mdm"][feature_idx]]
        results[feature_idx]["feature"]["sml"] = [
            (a, b.item()) for a, b in per_feature_logits["sml"][feature_idx]]

    results = {
        "results": results,
        "top_k": top_k
    }

    with open("./results.json", "w") as outfile:
        json.dump(results, outfile)

    return True
