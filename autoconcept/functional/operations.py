

class ScalerSum:

    def __init__(self, dim=-1):
        self.dim = dim

    def __call__(self, x):
        x_ = x / x.sum(dim=self.dim, keepdim=True)
        return x_
