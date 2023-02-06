import numpy as np
import torch
import torch.nn as nn


class GumbelSigmoid(nn.Module):
    """Implementation of Straight-Through version of Gumbel Sigmoid."""

    def __init__(
        self,
        t=0.5,
        eps=1e-20,
        hard=True,
        threshold=0.5,
        rate=1e-4,
        step=1000,
        min_val=0.5
    ):
        super().__init__()
        # gumbel config
        self.t = t
        self.eps = eps
        self.hard = hard
        self.threshold = threshold

        # annealing config
        self.rate = rate
        self.step = step
        self.min_val = min_val

    def forward(self, x, iteration=None):
        if iteration and iteration % self.step == 0:
            self.t = np.maximum(np.exp(-self.rate * iteration), self.min_val)
        t = self.t

        y = self._gumbel_sigmoid_sample(x, t)
        if not self.hard:
            return y
        indices = (y > self.threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(
            x, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0

        return y_hard - y.detach() + y

    def _gumbel_sigmoid_sample(self, x, t):
        temperature = t or self.t
        sample = self._sample_gumbel(x, x.device)
        gumbels = (x + sample) / temperature
        y_soft = gumbels.sigmoid()
        return y_soft

    def _sample_gumbel(self, x, device):
        gumbels = (-torch.empty_like(x, memory_format=torch.legacy_contiguous_format,
                   device=device).exponential_().log())
        return gumbels
