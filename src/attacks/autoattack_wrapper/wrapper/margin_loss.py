import torch

from secml.settings import SECML_PYTORCH_USE_CUDA

use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


class MarginLoss:
    def _adv_objective_function(self, x):
        scores = self.model(x)
        y0 = torch.empty(scores.shape[0], dtype=torch.long,
                         device="cuda" if use_cuda else "cpu")
        y0[:] = self._y0
        u = torch.arange(x.shape[0])
        y_corr = scores[u, y0].clone()
        scores[u, y0] = -float('inf')
        y_others = scores.max(dim=-1)[0]
        return y_corr - y_others
