import torch
import torch.nn as nn
import torchvision.models as models
from models.feature_extractors.base import BaseFeatureExtractor


class PneumoniaCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        n_features = self.network.fc.in_features
        self.network.fc = nn.Linear(n_features, 1)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


class DenseNet121(nn.Module):
    def __init__(self, num_classes, is_trained=True):
        super().__init__()
        self.net = models.densenet121(pretrained=is_trained)
        kernel_count = self.net.classifier.in_features
        self.net.classifier = nn.Sequential(
            nn.Linear(kernel_count, num_classes), nn.Sigmoid())

    def forward(self, inputs):
        return self.net(inputs)


class PretrainedResnetFeatureExtractor(BaseFeatureExtractor):

    def __init__(
        self,
        out_features=512,
    ):
        super().__init__()

        self.backbone_name = "resnet50"
        self.out_features = out_features

        self.__build()

    def __build(self):
        try:
            constructor = getattr(models, self.backbone_name)
        except AttributeError:
            raise ValueError(
                f"Backbone '{self.backbone_name}' is not supported!")

        backbone = PneumoniaCnnModel()
        backbone.load_state_dict(torch.load(
            "data/chest-x-ray-resnet50-model.pth"))

        backbone = backbone.network
        if hasattr(backbone, "fc"):
            if backbone.fc.out_features != self.out_features:
                backbone.fc = nn.Linear(
                    backbone.fc.in_features, self.out_features)

        self.main = backbone

    def forward(self, x):
        out = self.main(x)
        if hasattr(out, 'logits'):
            out = out.logits
        if type(out) is tuple:
            out = out[0]
        return out


class PretrainedDensenetFeatureExtractor(BaseFeatureExtractor):

    def __init__(
        self,
        out_features=512,
    ):
        super().__init__()
        self.out_features = out_features

        self.__build()

    def __build(self):
        backbone = DenseNet121(14)
        backbone.load_state_dict(torch.load(
            "data/densenet.pth")["model"])

        backbone = backbone.net
        if hasattr(backbone, "classifier"):
            if backbone.classifier[0].out_features != self.out_features:
                backbone.classifier = nn.Linear(
                    backbone.classifier[0].in_features, self.out_features)

        self.main = backbone

    def forward(self, x):
        out = self.main(x)
        if hasattr(out, 'logits'):
            out = out.logits
        if type(out) is tuple:
            out = out[0]
        return out


if __name__ == "__main__":
    feature_extractor = PretrainedDensenetFeatureExtractor(out_features=30)
    x = torch.ones(2, 3, 299, 299)
    y = feature_extractor(x)
    print(y.shape)
