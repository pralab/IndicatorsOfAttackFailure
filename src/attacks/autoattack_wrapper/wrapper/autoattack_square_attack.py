from .c_attack_evasion_autoattack import CAttackEvasionAutoAttack
from .margin_loss import MarginLoss
from ..autoattack.square import SquareAttack


class CAutoAttackSquare(CAttackEvasionAutoAttack, MarginLoss):
	__class_type = 'e-autoattack-square'

	def __init__(self, classifier, distance="linf", dmax=None,
				 version="standard", seed=None, n_queries=5000, n_restarts=1,
				 x_shape=None, y_target=None):
		"""

		Parameters
		----------
		classifier :
		distance :
		dmax :
		version :
		seed :
		n_queries :
		n_restarts :
		x_shape :
		"""
		super(CAutoAttackSquare, self).__init__(
			classifier, distance=distance, dmax=dmax)

		if version == "standard":
			self.attack = SquareAttack(self.model, norm=self.norm,
									   n_queries=5000, eps=dmax, n_restarts=1,
									   seed=seed, resc_schedule=False,
									   x_shape=x_shape)
		elif version == "plus":
			self.attack = SquareAttack(self.model, norm=self.norm,
									   n_queries=5000, eps=dmax, n_restarts=1,
									   seed=seed, resc_schedule=False,
									   x_shape=x_shape)
		elif version == "rand":
			raise ValueError("This attack is not included in rand "
							 "AutoAttack evaluation")
		elif version == "custom":
			self.attack = SquareAttack(self.model, norm=self.norm,
									   n_queries=n_queries, eps=dmax,
									   n_restarts=n_restarts, seed=seed,
									   resc_schedule=False, x_shape=x_shape)

	def _run(self, x, y, x_init=None):
		out, _ = super(CAutoAttackSquare, self)._run(x, y, x_init)
		self._f_seq = self.objective_function(self.x_seq)
		f_opt = self.objective_function(out)
		return out, f_opt
