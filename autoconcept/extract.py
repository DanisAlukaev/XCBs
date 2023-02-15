
import hydra
import numpy as np
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

    checkpoint_path = "/home/danis/Projects/AlphaCaption/AutoConceptBottleneck/autoconcept/outputs/2023-02-14/18-07-54/lightning_logs/version_0/checkpoints/last.ckpt"
    target_class = get_class(cfg.model._target_)
    main = instantiate(cfg.model.main)
    inference = target_class.load_from_checkpoint(
        checkpoint_path, main=main).cuda()

    n_concepts = len(inference.main.concept_extractor.encoders)
    distributions = [np.zeros(vocab_size) for _ in range(n_concepts)]
    n_tokens = np.zeros(vocab_size)

    for batch in tqdm(train_loader):
        indices = batch["indices"].cuda()
        for encoder_id in range(n_concepts):
            _, scores = inference.main.concept_extractor.encoders[encoder_id](
                indices, None)
            scores = scores.squeeze()
            scores_np = scores.cpu().detach().numpy()

            for sample_id in range(0, indices.shape[0]):
                indices_np = indices[sample_id].cpu().detach().numpy()
                scores_np_prev = distributions[encoder_id][indices_np]
                np.put(distributions[encoder_id], indices_np,
                       scores_np[sample_id] + scores_np_prev)

                if encoder_id == n_concepts - 1:
                    n_tokens[indices_np] += 1

    distributions = np.array(distributions) / n_tokens

    for i in range(n_concepts):
        print(f"Concept #{i}")
        ids = (-distributions[i]).argsort()[:10]
        scores = distributions[i][ids]
        itos_map = dm.dataloader_kwargs['collate_fn'].vocabulary.vocab.get_itos(
        )
        tokens = [itos_map[id] for id in ids]
        print(list(zip(tokens, scores)))
        print()

    # batch = next(iter(train_loader))
    # _, scores = inference.main.concept_extractor.encoders[0](batch["indices"].cuda(), None)
    # scores = scores.squeeze()
    # print(batch["indices"][0])
    # print(torch.sum(scores[0], axis=0))

    return None


if __name__ == "__main__":
    main()
