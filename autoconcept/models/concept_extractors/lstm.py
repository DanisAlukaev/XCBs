import torch
import torch.nn as nn
from models.concept_extractors.base import BaseConceptExtractor


class LSTMConceptExtractor(BaseConceptExtractor):

    def __init__(
        self,
        vocab_size=None,
        embed_dim=100,
        hidden_dim=100,
        n_layers=1,
        out_features=300,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_features = out_features

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, out_features)

    def forward(self, input_ids):
        device = input_ids.device
        x = self.embedding(input_ids)
        hidden, carry = torch.zeros(self.n_layers, len(input_ids), self.hidden_dim, device=device), torch.zeros(
            self.n_layers, len(input_ids), self.hidden_dim, device=device)
        features, (hidden, carry) = self.lstm(x, (hidden, carry))
        concepts = self.linear(features[:, -1])
        return concepts


if __name__ == "__main__":
    concept_extractor = LSTMConceptExtractor(
        vocab_size=10,
        embed_dim=100,
        hidden_dim=100,
        n_layers=1,
        n_concepts=300,
        activation=nn.ReLU(),
    )
    x = torch.randint(0, 10, (2, 15))
    y = concept_extractor(x)
    print(y.shape)
