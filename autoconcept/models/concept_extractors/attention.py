import torch
import torch.nn as nn
from models.concept_extractors.base import BaseConceptExtractor


class ConceptExtractorSimplifiedAttention(BaseConceptExtractor):

    def __init__(
        self,
        n_concepts=320,
        embed_dim=300
    ):
        super().__init__()

        self.n_concepts = n_concepts
        self.embed_dim = embed_dim

        self.queries = nn.Embedding(n_concepts, embed_dim)
        self.dummy = nn.Embedding(1, embed_dim)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, keys):
        batch_size = keys.shape[0]

        dummys = torch.stack([self.dummy.weight] * batch_size)
        queries = self.queries.weight
        _keys = torch.hstack([dummys, keys])
        attention = torch.matmul(queries, _keys.transpose(1, 2))

        ngram_probability = self.softmax(attention)
        concept_probs = 1 - ngram_probability[:, :, 0]

        return concept_probs
