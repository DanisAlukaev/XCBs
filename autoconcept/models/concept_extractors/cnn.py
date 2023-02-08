import torch
import torch.nn as nn
import torch.nn.functional as F
from models.concept_extractors.base import BaseConceptExtractor


class TextualCNN(nn.Module):

    def __init__(
        self,
        embed_dim=100,
        n_filters=512,
        filter_size=5,
        activation=nn.ReLU(),
        pooling_type="filter-wise"
    ):
        super().__init__()

        assert pooling_type in ["filter-wise", "overall"]

        self.embed_dim = embed_dim
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.activation = activation
        self.pooling_type = pooling_type

        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=n_filters,
            kernel_size=filter_size,
        )

    def forward(self, x):
        logits = self.conv(x)
        features = self.activation(logits)

        if self.pooling_type == "overall":
            features = features.view(x.shape[0], 1, -1)

        pool = F.max_pool1d(features, kernel_size=features.shape[-1])
        out = pool.view(x.shape[0], -1)
        return out


class ConceptExtractorSingleCNN(BaseConceptExtractor):

    def __init__(
        self,
        vocab_size=None,
        embed_dim=100,
        n_filters=512,
        filter_size=5,
        out_features=300,
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.out_features = out_features
        self.activation = activation

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cnn = TextualCNN(
            embed_dim=embed_dim,
            n_filters=n_filters,
            filter_size=filter_size,
            activation=activation,
            pooling_type="filter-wise"
        )

        self.linear = nn.Linear(n_filters, out_features)

    def forward(self, input_ids):
        x = self.embedding(input_ids).permute(0, 2, 1)
        features = self.cnn(x)
        concept_logits = self.linear(features)

        return concept_logits


class ConceptExtractorMultipleCNN(nn.Module):

    def __init__(
        self,
        vocab_size=None,
        embed_dim=100,
        n_filters=32,
        filter_size=5,
        out_features=300,
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.out_features = out_features
        self.activation = activation

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cnns = nn.ModuleList([
            TextualCNN(
                embed_dim=embed_dim,
                n_filters=n_filters,
                filter_size=filter_size,
                activation=activation,
                pooling_type="overall")
            for _ in range(out_features)
        ])

    def forward(self, input_ids):
        x = self.embedding(input_ids).permute(0, 2, 1)
        features = list()
        for _, cnn in enumerate(self.cnns):
            features.append(cnn(x))
        features = torch.stack(features, dim=1).squeeze(-1)
        return features
