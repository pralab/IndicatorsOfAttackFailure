from typing import Callable, TypeVar
import eagerpy as ep
import foolbox as fb
import torch
from foolbox import Model, Misclassification
from foolbox.attacks import LinfProjectedGradientDescentAttack
from numpy import NaN
from secml.adv.attacks import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.ce_loss import CELoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor, \
    as_carray
from secml.array import CArray


T = TypeVar("T")


class MisclassificationAdaptive(Misclassification):
    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs

        classes = outputs_.argmax(axis=-1)
        assert classes.shape == self.labels.shape
        is_adv = (classes != self.labels).logical_and(
            classes != outputs_.shape[-1] - 1)
        return restore_type(is_adv)


class PGDAdaptive(LinfProjectedGradientDescentAttack):
    def get_loss_fn(
            self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            rows = range(inputs.shape[0])
            logits = model(inputs)
            rej = logits.argmax(axis=-1).item() == logits.shape[-1] - 1
            c_minimize = ep.ones_like(labels) * 10 if rej else labels  # labels
            c_maximize = best_other_classes(logits, labels)
            loss = (logits[rows, c_maximize] - logits[rows, c_minimize]).sum()
            return loss

        return loss_fn


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    exclude = ep.ones_like(exclude) * -1
    other_logits = other_logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)


class CAttackEvasionFoolboxAdaptive(CAttackEvasionFoolbox):
    def _run(self, x, y, x_init=None):
        self.f_model.reset()
        if self.y_target is None:
            criterion = MisclassificationAdaptive(
                as_tensor(y.ravel().astype('int64')))
        else:
            criterion = fb.criteria.TargetedMisclassification(
                torch.tensor([self.y_target]))

        x_t = as_tensor(x, requires_grad=False)
        advx, clipped, is_adv = self.attack(
            self.f_model, x_t, criterion, epsilons=self.epsilon)

        if isinstance(clipped, list):
            if len(clipped) == 1:
                clipped = x[0]
            else:
                raise ValueError(
                    "This attack is returning a list. Please,"
                    "use a single value of epsilon.")

        # f_opt is computed only in class-specific wrappers
        f_opt = NaN

        self._last_f_eval = self.f_model.f_eval
        self._last_grad_eval = self.f_model.grad_eval
        path = self.f_model.x_path
        self._x_seq = CArray(path.numpy())

        # reset again to clean cached data
        self.f_model.reset()
        return as_carray(clipped), f_opt


class CFoolboxPGDLinfAdaptive(CELoss, CAttackEvasionFoolboxAdaptive):
    __class_type = 'e-foolbox-pgd-linf-adaptive'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2, distance='l2',
                 rel_stepsize=0.025, abs_stepsize=None, steps=50,
                 random_start=True):
        super(CFoolboxPGDLinfAdaptive, self).__init__(
            classifier, y_target,
            lb=lb, ub=ub,
            fb_attack_class=PGDAdaptive,
            epsilons=epsilons,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start)

        self._x0 = None
        self._y0 = None
        self.distance = distance

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxPGDLinfAdaptive, self)._run(x, y, x_init)
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(out)
        return out, f_opt
