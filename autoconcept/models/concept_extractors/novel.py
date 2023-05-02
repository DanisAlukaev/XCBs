import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.concept_extractors.base import BaseConceptExtractor
from models.predictors.mlp import MLPPredictor


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class ConceptExtractorAttention(BaseConceptExtractor):

    def __init__(
        self,
        vocab_size=None,
        embed_dim=100,
        out_features=512,
        activation=nn.ReLU(),
        src_pad_idx=0,
        max_len=350,
        dropout=0.,
        device="cuda",
        use_slot_norm=False,
        norm_fn1=F.softmax,
        norm_fn2=F.softmax,
        use_position_encoding=False,
        eps=1e-7,
        regularize_distance=True,
        regularize_distance_apply_p=1.0,
        mlp_depth=1,
        use_dummy_attention=True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.out_features = out_features
        self.activation = activation
        self.src_pad_idx = src_pad_idx
        self.max_len = max_len
        self.dropout = dropout
        self.device = device
        self.use_slot_norm = use_slot_norm
        self.norm_fn1 = norm_fn1
        self.norm_fn2 = norm_fn2
        self.use_position_encoding = use_position_encoding
        self.eps = eps
        self.regularize_distance = regularize_distance
        self.regularize_distance_apply_p = regularize_distance_apply_p
        self.mlp_depth = mlp_depth
        self.use_dummy_attention = use_dummy_attention

        self.mlp_layers = [self.embed_dim] * (mlp_depth - 1) + [1]

        # if use_dummy_attention:
        # self.mlp_layers[0] += 1

        self.values_w = nn.Linear(embed_dim, embed_dim)
        self.keys_w = nn.Linear(embed_dim, embed_dim)

        queries_n = out_features
        if use_slot_norm:
            dummy_query_n = 1
            queries_n += dummy_query_n
        self.queries_w = nn.Embedding(queries_n, embed_dim)

        self.word_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=self.src_pad_idx
        )

        dummy_tokens_n = queries_n
        self.dummy_tokens = nn.Embedding(dummy_tokens_n, embed_dim)

        self.position_embedding = nn.Embedding(max_len, embed_dim)
        if use_position_encoding:
            self.position_embedding = PositionalEncoding(
                embed_dim=embed_dim,
                max_len=max_len
            )

        self.mlps = nn.ModuleList([
            MLPPredictor(
                layers=self.mlp_layers,
                activation=nn.ReLU(),
                use_batch_norm=True,
                use_dropout=False,
                use_layer_norm=False
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
        concept_logits, concept_semantics = list(), list()

        N, seq_length = input_ids.shape
        mask = self.make_src_mask(input_ids)

        word_embedding = self.word_embedding(input_ids)
        if self.use_position_encoding:
            input_embedding = self.position_embedding(word_embedding)
        else:
            positions = torch.arange(0, seq_length)
            positions = positions.expand(N, seq_length).to(self.device)
            position_embedding = self.position_embedding(positions)
            input_embedding = word_embedding + position_embedding

        input_embedding = self.dropout(input_embedding)

        values = self.values_w(input_embedding)
        keys = self.keys_w(input_embedding)
        queries = self.queries_w.weight

        dummy_tokens = self.dummy_tokens.weight

        norm_factor = (self.embed_dim ** (1 / 2))
        attn_logits = torch.matmul(
            queries,
            keys.transpose(-2, -1)
        )
        attn_dummy_logits = torch.matmul(
            queries,
            dummy_tokens.transpose(-2, -1)
        ).unsqueeze(dim=0)

        attn_logits = attn_logits / norm_factor
        attn_dummy_logits = attn_dummy_logits / norm_factor

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        if self.use_slot_norm:
            scores = self.norm_fn1(attn_logits)
            if self.use_dummy_attention:
                # scores_max, _ = torch.max(scores, 2)
                # scores_mean = scores.mean(dim=-1)

                # scores_dummy = 1 - (scores_max + scores_mean)

                scores_dummy = 1 - torch.norm(scores, dim=-1)
                scores_dummy = torch.nn.functional.relu(scores_dummy)

                print(scores_dummy)

                scores_dummy = scores_dummy.unsqueeze(-1)

                # scores_dummy = self.norm_fn1(attn_dummy_logits)
                # scores_dummy = torch.diagonal(scores_dummy, 0)
                # scores_dummy = scores_dummy.expand(N, -1, -1)

                scores = scores.masked_fill(mask == 0, 0)
                scores = torch.cat((scores, scores_dummy), dim=2)

            scores = scores + self.eps
            scores = self.norm_fn2(scores)

            # TODO: use entire sequence with dummy tokens (or add it as additional feature)
            scores_aux = scores

            if self.use_dummy_attention:
                scores = scores[:, :, :seq_length]
        else:
            attn_dummy_logits = torch.diagonal(attn_dummy_logits, 0)
            attn_dummy_logits = attn_dummy_logits.expand(N, -1, -1)
            attn_logits = torch.cat((attn_logits, attn_dummy_logits), dim=2)

            attn_logits += self.eps

            scores = self.norm_fn1(attn_logits)

            scores_aux = scores
            scores_dummy = scores[:, :, -1].unsqueeze(-1)
            scores = scores[:, :, :seq_length]

        semantic = torch.matmul(scores, values)

        for idx, mlp in enumerate(self.mlps):
            concept_semantic = semantic[:, idx, :]
            if self.use_dummy_attention:
                score_dummy = scores_dummy[:, idx, :]

                # TODO: use only one dummy embedding
                # dummy_embedding = dummy_tokens[idx]
                dummy_embedding = dummy_tokens[0]

                concept_semantic = concept_semantic + score_dummy * dummy_embedding

                # concept_semantic = torch.cat(
                #     (concept_semantic, score_dummy), dim=1)

            concept_semantics.append(concept_semantic)

            concept_logit = mlp(concept_semantic)

            concept_logits.append(concept_logit)

        concept_logits = torch.stack(concept_logits, dim=1).squeeze(-1)
        out_dict = dict(
            concept_logits=concept_logits,
            scores=scores,
            scores_aux=scores_aux
        )

        if self.regularize_distance:
            similarities = list()
            for idx in range(len(concept_semantics)):
                for jdx in range(idx + 1, len(concept_semantics)):

                    if random.random() < self.regularize_distance_apply_p:
                        similarity = self.cosine_sim(
                            concept_semantics[idx],
                            concept_semantics[jdx]
                        ).abs()
                        similarities.append(similarity)

            avg_similarity = torch.cat(similarities).mean()
            out_dict["loss_dist"] = avg_similarity

        return out_dict
