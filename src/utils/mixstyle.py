import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle: Feature-level domain generalization layer.

    Mixes feature statistics (mean, std) across samples within a batch
    to simulate domain shift during training. Forces the model to learn
    domain-invariant features for cross-variety cotton leaf diagnosis.

    Reference: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        if torch.rand(1).item() > self.p:
            return x

        batch_size = x.size(0)
        if batch_size < 2:
            return x

        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], unbiased=False, keepdim=True)

        perm = torch.randperm(batch_size, device=x.device)

        mean_perm = mean[perm]
        var_perm = var[perm]

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size,)).to(x.device)
        lam = lam.view(batch_size, 1, 1, 1)

        mean_mixed = lam * mean + (1 - lam) * mean_perm
        var_mixed = lam * var + (1 - lam) * var_perm

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_mixed = x_norm * torch.sqrt(var_mixed + self.eps) + mean_mixed

        return x_mixed
