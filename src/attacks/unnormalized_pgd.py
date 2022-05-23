from secml.adv.attacks.evasion.foolbox.c_attack_evasion_foolbox import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.ce_loss import CELoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor


class CPGDLInfUnnormalized(CELoss, CAttackEvasionFoolbox):
    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0, epsilons=0.2,
                 rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True):
        attack = UnnormalizedPGD
        super(CPGDLInfUnnormalized, self).__init__(classifier, y_target,
                                                   lb=lb, ub=ub,
                                                   fb_attack_class=attack,
                                                   epsilons=epsilons,
                                                   rel_stepsize=rel_stepsize,
                                                   abs_stepsize=abs_stepsize,
                                                   steps=steps,
                                                   random_start=random_start)
        self._x0 = None
        self._y0 = None
        self.distance = 'linf'

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CPGDLInfUnnormalized, self)._run(x, y, x_init)
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(out)
        return out, f_opt


from foolbox.attacks import LinfProjectedGradientDescentAttack


class UnnormalizedPGD(LinfProjectedGradientDescentAttack):
    # pass
    def normalize(
            self, gradients, *, x, bounds):
        return gradients
