import torch
import torch.nn as nn


class KullbackLeiblerDivergenceLoss(nn.Module):

    def __init__(
        self,
        eps=1e-7,
        mode="native"
    ):
        super().__init__()
        self.implemented = {
            "bernoulli": self._bernoulli_kl,
            "direct": self._direct_kl,
            "native": self._native_kl,
        }
        assert mode in self.implemented.keys(
        ), f"Mode '{mode}' is not supported!"
        self.eps = eps
        self.mode = mode
        self._kl_loss = nn.KLDivLoss()

    def forward(self, P, Q):
        # P, Q = P + self.eps, Q + self.eps
        P = torch.clamp(P, min=0.0+self.eps, max=1.0-self.eps)
        Q = torch.clamp(Q, min=0.0+self.eps, max=1.0-self.eps)
        loss = self.implemented[self.mode](P, Q)
        return loss

    def _bernoulli_kl(self, P, Q):
        _P = torch.distributions.Bernoulli(probs=P)
        _Q = torch.distributions.Bernoulli(probs=Q)
        loss_c = torch.distributions.kl_divergence(_P, _Q).mean()
        return loss_c

    def _direct_kl(self, P, Q):
        kl_div = (P * torch.log(P / Q)).flatten()
        loss_c = kl_div.mean()
        return loss_c

    def _native_kl(self, Q, P):
        # Q = texutal, P = visual
        loss = (Q * torch.log(Q / P) + (1 - Q) *
                torch.log((1 - Q) / (1 - P))).mean()
        return loss
