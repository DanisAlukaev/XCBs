import torch
import torch.nn as nn


class KullbackLeiblerDivergenceLoss(nn.Module):

    def __init__(
        self,
        eps=1e-7,
    ):
        super().__init__()
        self.eps = eps

    def forward(self, P, Q):
        P = torch.clamp(P, min=0.0+self.eps, max=1.0-self.eps)
        Q = torch.clamp(Q, min=0.0+self.eps, max=1.0-self.eps)
        # print("P: ", P.min(), P.max())
        # print("Q: ", Q.min(), Q.max())
        # if torch.isnan(Q).any():
        #     print("Q (entire): ", Q)
        loss = (P * torch.log(P / Q) + (1 - P) *
                torch.log((1 - P) / (1 - Q))).mean()
        return loss


class JensenShannonDivergenceLoss(nn.Module):

    def __init__(
        self,
        eps=1e-7,
    ):
        super().__init__()
        self.eps = eps
        self.kl_div = KullbackLeiblerDivergenceLoss(eps=eps)

    def forward(self, P, Q):
        M = 0.5 * (P + Q)
        loss = 0.5 * self.kl_div(P, M) + 0.5 * self.kl_div(Q, M)
        return loss
