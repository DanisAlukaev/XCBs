import torch
import torch.nn as nn


def kullback_leibler_divergence(P, Q, eps=1e-7):
    P = torch.clamp(P, min=0.0+eps, max=1.0-eps)
    Q = torch.clamp(Q, min=0.0+eps, max=1.0-eps)
    dist = (P * torch.log(P / Q) + (1 - P) *
            torch.log((1 - P) / (1 - Q))).mean()
    return dist


class KullbackLeiblerDivergenceLoss(nn.Module):

    def __init__(
        self,
        eps=1e-7,
    ):
        super().__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits_P, logits_Q):
        P = self.sigmoid(logits_P)
        Q = self.sigmoid(logits_Q)

        loss = kullback_leibler_divergence(P, Q, self.eps)
        return loss


class JensenShannonDivergenceLoss(nn.Module):

    def __init__(
        self,
        eps=1e-7,
    ):
        super().__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits_P, logits_Q):
        P = self.sigmoid(logits_P)
        Q = self.sigmoid(logits_Q)
        M = 0.5 * (P + Q)

        loss = 0.5 * kullback_leibler_divergence(
            P, M, self.eps) + 0.5 * kullback_leibler_divergence(Q, M, self.eps)
        return loss
