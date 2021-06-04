from .c_attack_evasion_autoattack import CAttackEvasionAutoAttack
from ..autoattack.fab_pt import FABAttack_PT


class CAutoAttackFAB(CAttackEvasionAutoAttack):
	__class_type = 'e-autoattack-fab'

	def __init__(self, classifier, distance="linf", dmax=None,
				 version="standard", seed=None, n_restarts=1, n_iter=100,
				 y_target=None):
		"""

		Parameters
		----------
		classifier :
		distance :
		dmax :
		version :
		seed :
		n_restarts :
		n_iter :
		"""
		super(CAutoAttackFAB, self).__init__(classifier, distance=distance,
											 dmax=dmax)

		if version == "standard":
			raise ValueError("This attack is not included in standard "
							 "AutoAttack evaluation")
		elif version == "plus":
			self.attack = FABAttack_PT(self.model, norm=self.norm,
									   n_restarts=5, n_iter=100, eps=dmax,
									   seed=seed, targeted=False)
		elif version == "rand":
			raise ValueError("This attack is not included in rand "
							 "AutoAttack evaluation")
		elif version == "custom":
			self.attack = FABAttack_PT(self.model, norm=self.norm, eps=dmax,
									   n_restarts=n_restarts, targeted=False,
									   n_iter=n_iter, seed=seed)


class CAutoAttackFABT(CAttackEvasionAutoAttack):
	__class_type = 'e-autoattack-fab-t'

	def __init__(self, classifier, distance="linf", dmax=None,
				 version="standard", seed=None, n_restarts=1, n_iter=100,
				 y_target=None):
		super(CAutoAttackFABT, self).__init__(classifier, distance=distance,
											  dmax=dmax)

		if version == "standard":
			self.attack = FABAttack_PT(self.model, norm=self.norm,
									   n_restarts=1, n_iter=100, eps=dmax,
									   seed=seed, targeted=True)
		elif version == "plus":
			self.attack = FABAttack_PT(self.model, norm=self.norm,
									   n_restarts=5, n_iter=100, eps=dmax,
									   seed=seed, targeted=True)
		elif version == "rand":
			raise ValueError("This attack is not included in rand "
							 "AutoAttack evaluation")
		elif version == "custom":
			self.attack = FABAttack_PT(self.model, norm=self.norm, eps=dmax,
									   n_restarts=n_restarts, n_iter=n_iter,
									   seed=seed, targeted=True)
