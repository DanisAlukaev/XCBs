import torch
import torch.nn as nn
from models.concept_extractors.base import BaseConceptExtractor
from models.predictors.mlp import MLPPredictor


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers=1,
        heads=1,
        device="cuda",
        forward_expansion=4,
        dropout=0.,
        max_length=350,
    ):

        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)
        out = self.dropout((self.word_embedding(
            x) + self.position_embedding(positions)))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class ConceptExtractorAttention(BaseConceptExtractor):

    def __init__(
        self,
        vocab_size=None,
        embed_dim=100,
        out_features=300,
        device="cuda",
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.out_features = out_features
        self.activation = activation

        self.encoders = nn.ModuleList([
            TransformerEncoder(
                src_vocab_size=vocab_size,
                embed_size=embed_dim,
                num_layers=1,
                heads=1,
                device=device,
                forward_expansion=4,
                dropout=0.,
                max_length=350)
            for _ in range(out_features)
        ])

        self.mlps = nn.ModuleList([
            MLPPredictor(
                layers=[embed_dim, 1],
                activation=nn.ReLU(),
                use_batch_norm=True,
                use_dropout=False,
            )
            for _ in range(out_features)
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        concepts = list()
        for _, (encoder, mlp) in enumerate(zip(self.encoders, self.mlps)):
            embedding = encoder(input_ids, None).mean(dim=1)
            concepts.append(mlp(embedding))
        concept_logits = torch.stack(concepts, dim=1).squeeze(-1)
        concept_probs = self.sigmoid(concept_logits)

        return concept_logits
