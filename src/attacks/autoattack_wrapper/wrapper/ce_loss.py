from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch

from secml.array import CArray
from secml.settings import SECML_PYTORCH_USE_CUDA

from src.attacks.autoattack_wrapper.wrapper.secml_autoattack_autograd import as_tensor

use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


class CELossUntargeted:
    def _adv_objective_function(self, x):
        scores = self.model(x)
        y0 = torch.empty(scores.shape[0], dtype=torch.long,
                         device="cuda" if use_cuda else "cpu")
        if isinstance(self._y0, CArray):
            labels = as_tensor(self._y0)
        else:
            labels = self._y0
        y0[:] = labels
        loss = CrossEntropyLoss(reduce=False, reduction='none')
        y0[y0 == -1] = 10
        return -loss(scores, y0)


class CELossTargeted:
    def _adv_objective_function(self, x):
        scores = self.model(x)
        if self.attack.y_target is None:
            y0 = torch.empty(scores.shape[0], dtype=torch.long,
                             device="cuda" if use_cuda else "cpu")
            y0[:] = self._y0
            loss = CrossEntropyLoss(reduce=False, reduction='none')
            return loss(scores, y0)
        else:
            y_target = torch.empty(scores.shape[0], dtype=torch.long,
                             device="cuda" if use_cuda else "cpu")
            y_target[:] = self.attack.y_target
            total_loss = -1. * F.cross_entropy(scores, y_target, reduction='none')
            return total_loss
