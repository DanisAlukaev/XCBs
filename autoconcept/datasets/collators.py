import logging

import numpy as np
import torch
from datasets.embedders.fasttext_embedder import FastTextEmbedder
from datasets.utils import generate_ngrams, pad


class CollateEmulator:

    def __init__(self):
        pass

    def __call__(self, batch):
        batch_dict = dict()
        for sample in batch:
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])

        for key in batch_dict.keys():
            if not all(isinstance(x, torch.Tensor) for x in batch_dict[key]):
                continue
            batch_dict[key] = torch.stack(batch_dict[key])

        return batch_dict


class CollateBOW:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, batch):
        batch_dict = dict()
        indices = []
        bows = []
        max_len = -1
        for sample in batch:
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])
            tokens = self.vocabulary.tokenizer(sample["report"])
            if len(tokens) > max_len:
                max_len = len(tokens)
            id_ = self.vocabulary.vocab.lookup_indices(tokens)
            indices.append(id_)

            # TODO: compute on the fly
            bow = [0] * 8800
            for i in id_:
                bow[i] += 1
            bows.append(torch.tensor(bow).float())
        batch_dict["bow"] = bows
        for key in batch_dict.keys():
            if not all(isinstance(x, torch.Tensor) for x in batch_dict[key]):
                continue
            batch_dict[key] = torch.stack(batch_dict[key])

        return batch_dict


class CollateIndices:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        logging.info(
            f"Index for <pad>: {vocabulary.vocab.lookup_indices(['<pad>'])}")

    def __call__(self, batch):
        batch_dict = dict()
        indices = []
        max_len = -1
        for sample in batch:
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])
            tokens = self.vocabulary.tokenizer(sample["report"])
            if len(tokens) > max_len:
                max_len = len(tokens)
            indices.append(self.vocabulary.vocab.lookup_indices(tokens))

        _indices = list()
        for ids in indices:
            _indices.append(torch.tensor(
                ids + self.vocabulary.vocab.lookup_indices(["<pad>"]) * (max_len - len(ids))))

        batch_dict["indices"] = _indices

        for key in batch_dict.keys():
            if not all(isinstance(x, torch.Tensor) for x in batch_dict[key]):
                continue
            batch_dict[key] = torch.stack(batch_dict[key])

        return batch_dict


class CollateNgrams:

    def __init__(
        self,
        embedder=FastTextEmbedder(),
        n_token=5
    ):
        self.n_token = n_token
        self.embedder = embedder

    def __call__(self, batch):
        batch_dict = dict()
        for sample in batch:
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])

        ngrams = generate_ngrams(batch_dict['report'], n=self.n_token)
        max_len = len(sorted(ngrams, key=lambda x: len(x))[-1])
        ngrams = pad(ngrams, max_len)
        batch_dict['ngrams'] = ngrams

        keys = list()
        for ngram_list in ngrams:
            batch_embedding = self.embedder.batch_embed_sequence(ngram_list)
            batch_embedding_torch = torch.from_numpy(
                np.array(batch_embedding, dtype=np.float32))
            keys.append(batch_embedding_torch)
        batch_dict['keys'] = keys

        for key in batch_dict.keys():
            if not all(isinstance(x, torch.Tensor) for x in batch_dict[key]):
                continue
            batch_dict[key] = torch.stack(batch_dict[key])
        return batch_dict


if __name__ == "__main__":
    CollateNgrams()
