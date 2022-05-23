from abc import ABCMeta

import torch
from numpy import NaN
from secml.adv.attacks import CAttackEvasion

from .secml_autoattack_autograd import AutoattackSecmlLayer, as_tensor, \
    as_carray

use_cuda = torch.cuda.is_available()


class CAttackEvasionAutoAttack(CAttackEvasion, metaclass=ABCMeta):
    """

    Parameters
    ----------
    classifier : CClassifier
        Trained secml classifier.
    distance : str
        Can be either 'linf', 'l2' or 'l1'
    dmax : float or None, optional
        The maximum allowed perturbation.
    """
    __super__ = 'CAttackEvasionAutoAttack'

    def __init__(self, classifier, distance="linf", dmax=None):

        super(CAttackEvasionAutoAttack, self).__init__(classifier=classifier)

        self.distance = distance
        self.norm = distance.capitalize()

        self.seed = 0  # TODO: insert as parameter
        self.device = "cuda" if use_cuda else "cpu"

        # wraps secml classifier in a pytorch layer
        self.model = AutoattackSecmlLayer(classifier)

        self._last_f_eval = None
        self._last_grad_eval = None

        self._n_classes = self.classifier.n_classes
        self._n_feats = self.classifier.n_features

        self.dmax = dmax

        self._x0 = None
        self._y0 = None

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)

        self.model.reset()

        x_t = as_tensor(x, requires_grad=False)
        y_t = as_tensor(y).flatten()
        advx = self.attack.perturb(x_t, y_t)

        # f_opt is computed only in class-specific wrappers
        f_opt = NaN

        self._last_f_eval = self.model.f_eval
        self._last_grad_eval = self.model.grad_eval
        path = self.model.x_path
        self._x_seq = CArray(path.cpu().detach().numpy())

        # reset again to clean cached data
        self.model.reset()
        out = as_carray(advx)
        return out, f_opt

    def objective_function(self, x):
        return as_carray(self._adv_objective_function(as_tensor(x)))

    def objective_function_gradient(self, x):
        x_t = as_tensor(x).detach()
        x_t.requires_grad_()
        loss = self._adv_objective_function(x_t)
        loss.sum().backward()
        gradient = x_t.grad
        return as_carray(gradient)

    @property
    def x_seq(self):
        return self._x_seq

    @property
    def f_eval(self):
        if self._last_f_eval is not None:
            return self._last_f_eval
        else:
            raise RuntimeError("Attack not run yet!")

    @property
    def grad_eval(self):
        if self._last_grad_eval is not None:
            return self._last_grad_eval
        else:
            raise RuntimeError("Attack not run yet!")
