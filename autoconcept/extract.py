
import json
from pprint import pprint

import hydra
import numpy as np
import torch
from helpers import set_seed
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path="config/conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    dm = instantiate(cfg.dataset)
    dm.setup()

    train_loader, test_loader, val_loader = dm.train_dataloader(
    ), dm.test_dataloader(), dm.val_dataloader()

    vocab_size = len(dm.dataloader_kwargs['collate_fn'].vocabulary.vocab)
    print(f"Vocab size: {vocab_size}")

    checkpoint_path = "/home/danis/Projects/AlphaCaption/AutoConceptBottleneck/autoconcept/outputs/2023-03-29/17-39-30/lightning_logs/version_0/checkpoints/last.ckpt"
    target_class = get_class(cfg.model._target_)
    main = instantiate(cfg.model.main)
    inference = target_class.load_from_checkpoint(
        checkpoint_path, main=main).cuda()

    n_concepts = len(inference.main.concept_extractor.encoders)

    # Textual
    distributions = [np.zeros(vocab_size) for _ in range(n_concepts)]
    n_tokens = np.zeros(vocab_size)

    for batch in tqdm(train_loader):
        indices = batch["indices"].cuda()
        N, seq_length = indices.shape

        for encoder_id in range(n_concepts):
            positions = torch.arange(0, seq_length).expand(
                N, seq_length).to(inference.main.concept_extractor.device)
            input_embedding = inference.main.concept_extractor.dropout((inference.main.concept_extractor.word_embedding(
                indices) + inference.main.concept_extractor.position_embedding(positions)))
            mask = inference.main.concept_extractor.make_src_mask(indices)

            _, scores = inference.main.concept_extractor.encoders[encoder_id](
                input_embedding, mask)
            scores = scores.squeeze()
            scores_np = scores.cpu().detach().numpy()
            for sample_id in range(0, indices.shape[0]):
                indices_np = indices[sample_id].cpu().detach().numpy()

                for idx_elem, index_np in enumerate(indices_np):
                    distributions[encoder_id][index_np] += scores_np[sample_id][idx_elem]

                if encoder_id == n_concepts - 1:
                    n_tokens[indices_np] += 1

    results = list()
    distributions = np.array(distributions) / n_tokens
    # print(distributions.shape)
    top_k = 10
    for i in range(n_concepts):
        print(f"Concept #{i}")
        ids = (-distributions[i]).argsort()[:top_k]
        scores = distributions[i][ids]
        itos_map = dm.dataloader_kwargs['collate_fn'].vocabulary.vocab.get_itos(
        )
        tokens = [itos_map[id] for id in ids]
        numbers = [n_tokens[id] for id in ids]
        print(list(zip(tokens, scores, numbers)))
        print()

        results.append(
            dict(
                concept=list(zip(tokens, scores, numbers))
            )
        )

    # Visual
    top_k = 15
    instance_exploration_lrg = [list() for _ in range(n_concepts)]
    instance_exploration_sml = [list() for _ in range(n_concepts)]
    for batch in tqdm(train_loader):
        images = batch["image"].cuda()
        filenames = batch["img_path"]
        logits = inference.main.feature_extractor(images)

        logits = logits.cpu().detach()

        topk_lrg = torch.topk(logits, k=top_k, dim=0, largest=True)
        topk_sml = torch.topk(logits, k=top_k, dim=0, largest=False)

        lg_max_lrg = topk_lrg.values.t()
        lg_max_sml = topk_sml.values.t()
        # lg_max = torch.cat((lg_max_lrg, lg_max_sml), 1)

        ids_lrg = topk_lrg.indices.t()
        ids_sml = topk_sml.indices.t()
        # ids = torch.cat((ids_lrg, ids_sml), 1)

        filenames_topk_lrg = np.array([filenames[id]
                                       for id in ids_lrg.flatten().tolist()])
        filenames_topk_lrg = filenames_topk_lrg.reshape(ids_lrg.shape)

        filenames_topk_sml = np.array([filenames[id]
                                       for id in ids_sml.flatten().tolist()])
        filenames_topk_sml = filenames_topk_sml.reshape(ids_sml.shape)

        pairs_lrg = [list(zip(filenames_topk_lrg[_], lg_max_lrg[_]))
                     for _ in range(n_concepts)]
        pairs_sml = [list(zip(filenames_topk_sml[_], lg_max_sml[_]))
                     for _ in range(n_concepts)]

        instance_exploration_lrg = [sorted(a + b, reverse=True, key=lambda x: abs(
            x[1]))[:top_k] for a, b in zip(instance_exploration_lrg, pairs_lrg)]
        instance_exploration_sml = [sorted(a + b, reverse=True, key=lambda x: abs(
            x[1]))[:top_k] for a, b in zip(instance_exploration_sml, pairs_sml)]

    instance_exploration = [
        a + b for a, b in zip(instance_exploration_lrg, instance_exploration_sml)]

    for i in range(len(instance_exploration)):
        print(f"Concept #{i}")
        pprint(instance_exploration[i])
        results[i]["feature"] = [(a, b.item())
                                 for a, b in instance_exploration[i]]
        print()

    with open("/home/danis/Projects/AlphaCaption/AutoConceptBottleneck/autoconcept/results.json", "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    main()
