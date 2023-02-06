import torch.nn as nn


class BaseFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError(
            f"Method of abstract class for '{type(self).__name__}' was used")

    def get_samples_per_feature(self):
        raise NotImplementedError(
            f"Method of abstract class for '{type(self).__name__}' was used")
