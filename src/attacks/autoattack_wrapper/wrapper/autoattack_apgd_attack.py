from .c_attack_evasion_autoattack import CAttackEvasionAutoAttack
from .ce_loss import CELossUntargeted
from .dlr_loss import DLRLossUntargeted
from ..autoattack.autopgd_base import APGDAttack


class CAutoAttackAPGDCE(CAttackEvasionAutoAttack, CELossUntargeted):
    __class_type = 'e-autoattack-apgd-ce'

    def __init__(self, classifier, distance="linf", dmax=None,
                 version="standard", seed=None, n_iter=100, n_restarts=1,
                 eot_iter=1, rho=.75, use_larger_eps=False, y_target=None):

        super(CAutoAttackAPGDCE, self).__init__(
            classifier, distance=distance, dmax=dmax)

        if version == "standard":
            self.attack = APGDAttack(self.model, n_iter=100, norm=self.norm,
                                     n_restarts=5 if self.norm == "L1" else 1,
                                     eps=dmax, seed=seed, loss="ce",
                                     eot_iter=1, rho=.75,
                                     use_largereps=self.norm == "L1")
        elif version == "plus":
            self.attack = APGDAttack(self.model, n_iter=100, norm=self.norm,
                                     n_restarts=5, eps=dmax, seed=seed,
                                     loss="ce", eot_iter=1, rho=.75,
                                     use_largereps=False)
        elif version == "rand":
            self.attack = APGDAttack(self.model, n_iter=100, norm=self.norm,
                                     n_restarts=1, eps=dmax, seed=seed,
                                     loss="ce", eot_iter=20, rho=.75,
                                     use_largereps=False)
        elif version == "custom":
            self.attack = APGDAttack(self.model, n_iter=n_iter, norm=self.norm,
                                     n_restarts=n_restarts, eps=dmax,
                                     seed=seed, loss="ce", eot_iter=eot_iter,
                                     rho=rho, use_largereps=use_larger_eps)

    def _run(self, x, y, x_init=None):
        out, _ = super(CAutoAttackAPGDCE, self)._run(x, y, x_init)
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(out)
        return out, f_opt


class CAutoAttackAPGDDLR(CAttackEvasionAutoAttack, DLRLossUntargeted):
    __class_type = 'e-autoattack-apgd-dlr'

    def __init__(self, classifier, distance="linf", dmax=None,
                 version="standard", seed=None, n_iter=100, n_restarts=1,
                 eot_iter=1, rho=.75, use_larger_eps=False, y_target=None):
        super(CAutoAttackAPGDDLR, self).__init__(
            classifier, distance=distance, dmax=dmax)

        if version == "standard":
            print("This attack is not included in standard "
                  "AutoAttack evaluation")
            self.attack = APGDAttack(self.model, n_iter=100, norm=self.norm,
                                     n_restarts=1, eps=dmax, seed=seed, loss="dlr",
                                     eot_iter=1, rho=.75,
                                     use_largereps=self.norm == "L1")
        elif version == "plus":
            self.attack = APGDAttack(self.model, n_iter=100, norm=self.norm,
                                     n_restarts=5, eps=dmax, seed=seed,
                                     loss="dlr", eot_iter=1, rho=.75,
                                     use_largereps=False)
        elif version == "rand":
            self.attack = APGDAttack(self.model, n_iter=100, norm=self.norm,
                                     n_restarts=1, eps=dmax, seed=seed,
                                     loss="dlr", eot_iter=20, rho=.75,
                                     use_largereps=False)
        elif version == "custom":
            self.attack = APGDAttack(self.model, n_iter=n_iter, norm=self.norm,
                                     n_restarts=n_restarts, eps=dmax,
                                     seed=seed, loss="dlr", eot_iter=eot_iter,
                                     rho=rho, use_largereps=use_larger_eps)

    def _run(self, x, y, x_init=None):
        out, _ = super(CAutoAttackAPGDDLR, self)._run(x, y, x_init)
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(out)
        return out, f_opt
