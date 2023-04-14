import torch
import torch.nn as nn
from models.predictors.base import BasePredictor


class MLPPredictor(BasePredictor):
    """Implementation of Multilayer Perceptron."""

    def __init__(
        self,
        layers=[512, 256, 128, 64, 1],
        activation=nn.ReLU(),
        use_batch_norm=True,
        use_dropout=False,
    ):
        super().__init__()

        self.layers = layers
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.main = self.__build(
            layers, activation, use_batch_norm, use_dropout)

    def __build(self, layers, activation, use_batch_norm, use_dropout):
        main = list()
        n_layers = len(layers)
        for layer_idx in range(n_layers - 1):
            in_dim = layers[layer_idx]
            out_dim = layers[layer_idx + 1]
            linear = nn.Linear(in_dim, out_dim)

            modules = [linear]
            if use_batch_norm:
                batch_norm = nn.BatchNorm1d(out_dim)
                modules.append(batch_norm)

            if layer_idx == n_layers - 2:
                main.extend(modules)
                continue

            modules.append(activation)

            if use_dropout:
                dropout = nn.Dropout(p=0.5)
                modules.append(dropout)

            main.extend(modules)

        main = nn.Sequential(*main)
        return main

    def forward(self, x):
        out = self.main(x)
        return out


if __name__ == "__main__":
    mlp = MLPPredictor()
    x = torch.ones(2, 512)
    y = mlp(x)
    print(y.shape)
