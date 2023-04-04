import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.concept_extractors.base import BaseConceptExtractor
from models.predictors.mlp import MLPPredictor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        values_w,
        keys_w,
        queries_w,
        idx,
        device,
        slot_norm,
        slot_norm_fn1,
        eps=1e-7,
    ):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.values_w = values_w
        self.keys_w = keys_w
        self.queries_w = queries_w
        self.idx = idx
        self.device = device
        self.slot_norm = slot_norm
        self.slot_norm_fn1 = slot_norm_fn1
        self.eps = eps

    def forward(self, input_embedding, mask):
        values = self.values_w(input_embedding)
        keys = self.keys_w(input_embedding)
        queries = self.queries_w.weight

        # print("Values: ", values.min(), values.max())
        # print("Keys", keys.min(), keys.max())

        attn_logits = torch.matmul(queries, keys.transpose(-2, -1))
        attn_logits = attn_logits / (self.embed_dim ** (1 / 2))
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        if not self.slot_norm:
            attention_concepts = F.softmax(attn_logits, dim=-1)
        else:
            attention_concepts = self.slot_norm_fn1(attn_logits)
            # print(attention_concepts)
            attention_concepts = attention_concepts + self.eps
            attention_concepts = attention_concepts / \
                attention_concepts.sum(dim=-1, keepdim=True)

        attention = attention_concepts[:, self.idx, :].unsqueeze(dim=1)
        # print("Attention", attention.min(), attention.max())
        out = torch.matmul(attention, values)
        # print("Out", out.min(), out.max())

        return out, attention


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        values_w,
        keys_w,
        queries_w,
        forward_expansion,
        idx,
        device,
        slot_norm,
        slot_norm_fn1
    ):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.device = device

        self.attention = Attention(
            embed_dim, values_w, keys_w, queries_w, idx, device, slot_norm, slot_norm_fn1)
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
        slot_norm=False,
        slot_norm_fn1=F.softmax,
        use_position_encoding=False,
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
        self.slot_norm = slot_norm
        self.slot_norm_fn1 = slot_norm_fn1
        self.use_position_encoding = use_position_encoding

        self.values_w = nn.Linear(embed_dim, embed_dim)
        self.keys_w = nn.Linear(embed_dim, embed_dim)

        if not slot_norm:
            self.queries_w = nn.Embedding(out_features, embed_dim)
        else:
            self.queries_w = nn.Embedding(out_features + 1, embed_dim)

        self.word_embedding = nn.Embedding(vocab_size, embed_dim)

        if use_position_encoding:
            self.position_embedding = PositionalEncoding(
                d_model=embed_dim, max_len=max_length)
        else:
            self.position_embedding = nn.Embedding(max_length, embed_dim)

        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=embed_dim,
                values_w=self.values_w,
                keys_w=self.keys_w,
                queries_w=self.queries_w,
                forward_expansion=forward_expansion,
                idx=idx,
                device=device,
                slot_norm=slot_norm,
                slot_norm_fn1=slot_norm_fn1
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
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        return src_mask.to(self.device)

    def forward(self, input_ids):
        N, seq_length = input_ids.shape
        concept_logits = list()
        embeddings = list()
        # print("Queries: ", self.queries_w.weight.min(), self.queries_w.weight.max())
        # print("Embeddings: ", self.word_embedding.weight.min(), self.word_embedding.weight.max())
        for idx, (encoder, mlp) in enumerate(zip(self.encoders, self.mlps)):
            weights_encoder = torch.cat([p.flatten()
                                        for p in encoder.parameters()])
            weights_mlp = torch.cat([p.flatten() for p in mlp.parameters()])
            # print(f"A-{idx}: ", weights_encoder.min(), weights_encoder.max())
            # print(f"M-{idx}: ", weights_mlp.min(), weights_mlp.max())

            # regular embedding + positional embedding
            word_embedding = self.word_embedding(input_ids)
            if self.use_position_encoding:
                input_embedding = self.position_embedding(word_embedding)
            else:
                positions = torch.arange(0, seq_length).expand(
                    N, seq_length).to(self.device)
                input_embedding = word_embedding + \
                    self.position_embedding(positions)

            input_embedding = self.dropout(input_embedding)
            mask = self.make_src_mask(input_ids)

            embedding, _ = encoder(input_embedding, mask)
            avg_semantic = embedding.squeeze(dim=1)  # 64, 100
            embeddings.append(avg_semantic)
            concept_logit = mlp(avg_semantic)

            concept_logits.append(concept_logit)
        concept_logits = torch.stack(concept_logits, dim=1).squeeze(-1)

        dists = list()
        for idx in range(len(embeddings)):
            for jdx in range(idx + 1, len(embeddings)):
                ea, eb = embeddings[idx], embeddings[jdx]
                dist = self.cosine_sim(ea, eb)
                dists.append(dist)
        avg_dist = torch.cat(dists).mean()
        return concept_logits, avg_dist
