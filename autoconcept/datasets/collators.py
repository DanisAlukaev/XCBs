import logging

import numpy as np
import torch
from datasets.embedders.abstract_embedder import Embedder
from datasets.embedders.fasttext_embedder import FastTextEmbedder
from datasets.utils import Vocabulary, generate_ngrams, pad


class CollateEmulator:
    """Emulation of a native collator in PyTorch."""

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        batch: list,
    ) -> dict:
        """Combines data together in batches. This implementation
        replicates default collate_fn in PyTorch and is likely to
        fail for samples of different len.

        :param batch: list of samples from dataset to batch over.
        :return: dict with batched samples.
        """
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
    """Collator for a Bag of Words representation."""

    def __init__(
        self,
        vocabulary: Vocabulary
    ) -> None:
        """Constructor of a Bag of Words collator.

        :param vocabulary: vocabulary of textual dataset.
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary.vocab)

    def __call__(
        self,
        batch: list,
    ) -> dict:
        """Combines data together in batches. In comparison with
        default collate function, this implementation generates
        additional keyword 'bow' with a Bag of Words representation
        of a caption.

        :param batch: list of samples from dataset to batch over.
        :return: dict with batched samples and 'bow' keyword.
        """
        batch_dict = dict()
        bows = list()

        for sample in batch:
            bow = [0.] * self.vocab_size
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])
            tokens = self.vocabulary.tokenizer(sample["report"])
            ids = self.vocabulary.vocab.lookup_indices(tokens)
            for id in ids:
                bow[id] += 1
            bows.append(torch.tensor(bow).float())
        batch_dict["bow"] = bows

        for key in batch_dict.keys():
            if not all(isinstance(x, torch.Tensor) for x in batch_dict[key]):
                continue
            batch_dict[key] = torch.stack(batch_dict[key])

        return batch_dict


class CollateIndices:

    def __init__(
        self,
        vocabulary: Vocabulary
    ) -> None:
        """Constructor of a vector of indices collator.

        :param vocabulary: vocabulary of textual dataset.
        """
        self.vocabulary = vocabulary

        self.pad_idx = vocabulary.vocab.lookup_indices(['<pad>'])
        logging.info(f"Index for <pad>: {self.pad_idx}")

    def __call__(
        self,
        batch: list,
    ) -> dict:
        """Combines data together in batches. In comparison with
        default collate function, this implementation generates
        additional keyword 'indices' with a vector of indices
        that can be used by a model to access word embeddings.

        :param batch: list of samples from dataset to batch over.
        :return: dict with batched samples and 'bow' keyword.
        """
        batch_dict = dict()
        indices, caption_lens = list(), list()

        for sample in batch:
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])
            tokens = self.vocabulary.tokenizer(sample["report"])
            caption_lens.append(len(tokens))
            indices.append(self.vocabulary.vocab.lookup_indices(tokens))

        max_len = max(caption_lens)
        indices_unilen = list()
        for ids in indices:
            pad_size = max_len - len(ids)
            pad_indices = self.pad_idx * pad_size
            indices_unilen.append(torch.tensor(ids + pad_indices))
        batch_dict["indices"] = indices_unilen

        for key in batch_dict.keys():
            if not all(isinstance(x, torch.Tensor) for x in batch_dict[key]):
                continue
            batch_dict[key] = torch.stack(batch_dict[key])

        return batch_dict


class CollateNgrams:

    def __init__(
        self,
        embedder: Embedder = FastTextEmbedder(),
        n_tokens: int = 5
    ) -> None:
        """Constructor of a collator for n-gram embeddings.

        Important note: this collator is currently obsolete!

        :param embedder: pre-trained textual embedder.
        :param n_tokens: size of n-gram
        """
        self.n_tokens = n_tokens
        self.embedder = embedder

    def __call__(
        self,
        batch: list,
    ) -> dict:
        """Combines data together in batches. In comparison with
        default collate function, this implementation generates
        additional keyword 'keys' with an embedding of n-gram
        obtained from pre-trained embedder.

        :param batch: list of samples from dataset to batch over.
        :return: dict with batched samples and 'bow' keyword.
        """
        batch_dict = dict()
        for sample in batch:
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])

        ngrams = generate_ngrams(batch_dict['report'], n=self.n_tokens)
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
