import torch

from secml.settings import SECML_PYTORCH_USE_CUDA

use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


class DLRLossUntargeted:
    def _adv_objective_function(self, x):
        scores = self.model(x)
        y0 = torch.empty(scores.shape[0], dtype=torch.long,
                         device="cuda" if use_cuda else "cpu")
        y0[:] = self._y0
        scores_sorted, ind_sorted = scores.sort(dim=1)
        ind = (ind_sorted[:, -1] == y0).float()
        u = torch.arange(scores.shape[0])
        return -(scores[u, y0] - scores_sorted[:, -2] * ind -
                 scores_sorted[:, -1] * (1. - ind)) / \
                (scores_sorted[:, -1] - scores_sorted[:, -3] + 1e-12)


class DLRLossTargeted:
    def _adv_objective_function(self, x):
        scores = self.model(x)
        y0 = torch.empty(scores.shape[0], dtype=torch.long,
                         device="cuda" if use_cuda else "cpu")
        y0[:] = self._y0
        if self.attack.y_target is None:
            scores_sorted, ind_sorted = scores.sort(dim=1)
            ind = (ind_sorted[:, -1] == y0).float()
            u = torch.arange(scores.shape[0])
            return -(scores[u, y0] - scores_sorted[:, -2] * ind -
                     scores_sorted[:, -1] * (1. - ind)) / \
                   (scores_sorted[:, -1] - scores_sorted[:, -3] + 1e-12)
        else:
            y_target = torch.empty(scores.shape[0], dtype=torch.long,
                             device="cuda" if use_cuda else "cpu")
            y_target[:] = self.attack.y_target
            scores_sorted, ind_sorted = scores.sort(dim=1)
            u = torch.arange(scores.shape[0])

            return -(scores[u, y0] - scores[u, y_target]) / \
                   (scores_sorted[:, -1] - .5 * (scores_sorted[:, -3] +
                                                 scores_sorted[:, -4]) + 1e-12)


class DLRLossUntargetedAdaptive:
    def _adv_objective_function(self, x):
        scores = self.model(x)
        y0 = torch.empty(scores.shape[0], dtype=torch.long,
                         device="cuda" if use_cuda else "cpu")
        y0[:] = self._y0
        rej = scores.argmax(dim=-1) == scores.shape[-1] - 1
        c_min = y0.clone()
        c_min[rej] = scores.shape[-1] - 1
        scores_sorted, ind_sorted = scores[:, :-1].sort(
            dim=1)  # remove reject class
        ind = (ind_sorted[:, -1] == y0).float()
        u = torch.arange(scores.shape[0])
        return -(scores[u, c_min] - scores_sorted[:,
                                      -2] * ind - scores_sorted[:, -1] * (
                         1. - ind)) / (
                           scores_sorted[:, -1] - scores_sorted[:, -3] + 1e-12)
