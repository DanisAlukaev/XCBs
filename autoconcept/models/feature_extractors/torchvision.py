import torch
import torch.nn as nn
import torchvision.models as models
from models.feature_extractors.base import BaseFeatureExtractor


class TorchvisionFeatureExtractor(BaseFeatureExtractor):

    def __init__(
        self,
        model="resnet18",
        weights="IMAGENET1K_V1",
        out_features=512,
    ):
        super().__init__()

        self.backbone_name = model
        self.weights = weights
        self.out_features = out_features

        self.__build()

    def __build(self):
        try:
            constructor = getattr(models, self.backbone_name)
        except AttributeError:
            raise ValueError(
                f"Backbone '{self.backbone_name}' is not supported!")

        backbone = constructor(weights=self.weights)
        # if not self.weights:
        #     for layer in backbone.children():
        #         if hasattr(layer, 'reset_parameters'):
        #             layer.reset_parameters()

        if hasattr(backbone, "fc"):
            if backbone.fc.out_features != self.out_features:
                backbone.fc = nn.Linear(
                    backbone.fc.in_features, self.out_features)

        if hasattr(backbone, "classifier"):
            if backbone.classifier.out_features != self.out_features:
                backbone.classifier = nn.Linear(
                    backbone.classifier.in_features, self.out_features)

        self.main = backbone

    def forward(self, x):
        out = self.main(x)
        if hasattr(out, 'logits'):
            out = out.logits
        if type(out) is tuple:
            out = out[0]
        return out


if __name__ == "__main__":
    feature_extractor = TorchvisionFeatureExtractor(
        model="inception_v3",
        weights="IMAGENET1K_V1"
    )
    x = torch.ones(2, 3, 299, 299)
    y = feature_extractor(x)
    print(y.shape)
