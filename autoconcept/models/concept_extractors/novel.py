import torch
import torch.nn as nn
from models.concept_extractors.base import BaseConceptExtractor
from models.predictors.mlp import MLPPredictor


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        values_w,
        keys_w,
        device,
    ):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.values_w = values_w
        self.keys_w = keys_w
        self.query_w = nn.Embedding(1, embed_dim)
        self.device = device

    def forward(self, input_embedding, mask):
        seq_len = input_embedding.shape[1]

        values = self.values_w(input_embedding)
        keys = self.keys_w(input_embedding)
        queries = torch.stack([self.query_w.weight] *
                              seq_len, dim=0).squeeze(dim=1)
        energy = torch.einsum("qd,nkd->nqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=2)
        out = torch.einsum("nql,nld->nqd", [attention, values])
        return out, attention


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        values_w,
        keys_w,
        forward_expansion,
        device
    ):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.values_w = values_w
        self.keys_w = keys_w
        self.device = device

        self.attention = Attention(embed_dim, values_w, keys_w, device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

    def forward(self, input_embedding, mask):
        attention, scores = self.attention(input_embedding, mask)

        x = self.norm1(attention)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out, scores


class ConceptExtractorAttention(BaseConceptExtractor):

    def __init__(
        self,
        vocab_size=None,
        embed_dim=100,
        out_features=512,
        activation=nn.ReLU(),
        src_pad_idx=0,
        max_length=350,
        dropout=0.,
        forward_expansion=4,
        device="cuda",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.out_features = out_features
        self.activation = activation
        self.src_pad_idx = src_pad_idx
        self.max_length = max_length
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.device = device

        self.values_w = nn.Linear(embed_dim, embed_dim)
        self.keys_w = nn.Linear(embed_dim, embed_dim)

        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=embed_dim,
                values_w=self.values_w,
                keys_w=self.keys_w,
                forward_expansion=forward_expansion,
                device=device,
            )
            for idx in range(out_features)
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
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        return src_mask.to(self.device)

    def forward(self, input_ids):
        N, seq_length = input_ids.shape
        concept_logits = list()
        for _, (encoder, mlp) in enumerate(zip(self.encoders, self.mlps)):
            # regular embedding + positional embedding
            positions = torch.arange(0, seq_length).expand(
                N, seq_length).to(self.device)
            input_embedding = self.dropout((self.word_embedding(
                input_ids) + self.position_embedding(positions)))
            mask = self.make_src_mask(input_ids)

            embedding, _ = encoder(input_embedding, mask)
            avg_semantic = embedding.mean(dim=1)
            concept_logit = mlp(avg_semantic)

            concept_logits.append(concept_logit)
        concept_logits = torch.stack(concept_logits, dim=1).squeeze(-1)
        return concept_logits
