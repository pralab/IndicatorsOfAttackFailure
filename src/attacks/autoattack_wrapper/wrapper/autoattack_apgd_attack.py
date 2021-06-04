from .c_attack_evasion_autoattack import CAttackEvasionAutoAttack
from .ce_loss import CELossUntargeted
from .dlr_loss import DLRLossUntargeted, DLRLossTargeted, \
	DLRLossUntargetedAdaptive
from ..autoattack.autoattack_adaptive import APGDAttackAdaptive
from ..autoattack.autopgd_base import APGDAttack, APGDAttack_targeted


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
			raise ValueError("This attack is not included in standard "
							 "AutoAttack evaluation")
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


class CAutoAttackAPGDT(CAttackEvasionAutoAttack, DLRLossTargeted):
	__class_type = 'e-autoattack-apgd-t'

	def __init__(self, classifier, distance="linf", dmax=None,
				 version="standard", seed=None, n_iter=100, n_restarts=1,
				 eot_iter=1, rho=.75, n_target_classes=9, use_larger_eps=False,
				 y_target=None):
		"""

		Parameters
		----------
		classifier :
		distance :
		dmax :
		version :
		seed :
		n_iter :
		n_restarts :
		eot_iter :
		rho :
		n_target_classes :
		use_larger_eps :
		"""
		super(CAutoAttackAPGDT, self).__init__(
			classifier, distance=distance, dmax=dmax)

		if version == "standard":
			self.attack = APGDAttack_targeted(self.model, n_iter=100,
											  norm=self.norm, n_restarts=1,
											  eps=dmax, seed=seed,
											  eot_iter=1, rho=.75,
											  n_target_classes=5 if
											  self.norm == "L1" else 9,
											  use_largereps=self.norm == "L1")
		elif version == "plus":
			self.attack = APGDAttack_targeted(self.model, n_iter=100,
											  norm=self.norm, n_restarts=1,
											  eps=dmax, seed=seed,
											  eot_iter=1, rho=.75,
											  n_target_classes=9,
											  use_largereps=False)
		elif version == "rand":
			raise ValueError("This attack is not included in rand "
							 "AutoAttack evaluation")
		elif version == "custom":
			self.attack = APGDAttack_targeted(self.model, n_iter=n_iter,
											  norm=self.norm,
											  n_restarts=n_restarts,
											  eps=dmax, seed=seed,
											  eot_iter=eot_iter, rho=rho,
											  use_largereps=use_larger_eps,
											  n_target_classes=n_target_classes
											  )

	def _run(self, x, y, x_init=None):
		out, _ = super(CAutoAttackAPGDT, self)._run(x, y, x_init)
		self._f_seq = self.objective_function(self.x_seq)
		f_opt = self.objective_function(out)
		return out, f_opt


class CAutoAttackAPGDDLRAdaptive(CAttackEvasionAutoAttack,
								 DLRLossUntargetedAdaptive):
	__class_type = 'e-autoattack-apgd-dlr-adaptive'

	def __init__(self, classifier, distance="linf", dmax=None,
				 version="standard", seed=None, n_iter=100, n_restarts=1,
				 eot_iter=1, rho=.75, use_larger_eps=False, y_target=None):
		super(CAutoAttackAPGDDLRAdaptive, self).__init__(
			classifier, distance=distance, dmax=dmax)

		if version == "standard":
			raise ValueError("This attack is not included in standard "
							 "AutoAttack evaluation")
		elif version == "plus":
			self.attack = APGDAttackAdaptive(
				self.model, n_iter=100, norm=self.norm, n_restarts=5, eps=dmax,
				seed=seed, loss="dlr", eot_iter=1, rho=.75,
				use_largereps=False)
		elif version == "rand":
			self.attack = APGDAttackAdaptive(
				self.model, n_iter=100, norm=self.norm, n_restarts=1, eps=dmax,
				seed=seed, loss="dlr", eot_iter=20, rho=.75,
				use_largereps=False)
		elif version == "custom":
			self.attack = APGDAttackAdaptive(
				self.model, n_iter=n_iter, norm=self.norm,
				n_restarts=n_restarts, eps=dmax, seed=seed, loss="dlr",
				eot_iter=eot_iter, rho=rho, use_largereps=use_larger_eps)

	def _run(self, x, y, x_init=None):
		out, _ = super(CAutoAttackAPGDDLRAdaptive, self)._run(x, y, x_init)
		self._f_seq = self.objective_function(self.x_seq)
		f_opt = self.objective_function(out)
		return out, f_opt
