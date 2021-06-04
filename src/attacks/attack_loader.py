from secml.adv.attacks import CFoolboxPGDLinf

from src.attacks.autoattack_wrapper import CAutoAttackAPGDCE, CAutoAttackAPGDDLR
from src.attacks.autoattack_wrapper.wrapper.autoattack_apgd_attack import CAutoAttackAPGDDLRAdaptive
from src.attacks.pgd_adaptive import CFoolboxPGDLinfAdaptive
from src.attacks.pgd_logits import CFoolboxLogitsPGD
from src.attacks.smoothed_pgd import CFoolboxAveragedPGD
from src.models.model_loader import check_model_id, load_model

ORIGINAL_ATTACKS = {
	0: {
		"attack": "CFoolboxPGDLinf",
		"model_id": 0,
		"attack_params": {
			"y_target": None,
			"epsilons": 0.031,
			"abs_stepsize": 0.01,
			"random_start": False,
			"steps": 50
		},
		"random_restarts": 5,
	},

	1: {
		"attack": "CFoolboxPGDLinf",
		"model_id": 1,
		"attack_params": {
			"y_target": None,
			"epsilons": 0.3,
			"abs_stepsize": 0.1,
			"random_start": False,
			"steps": 50
		},
		"random_restarts": 5,
	},

	2: {
		"attack": "CFoolboxPGDLinf",
		"model_id": 2,
		"attack_params": {
			"y_target": None,
			"epsilons": 0.01,
			"abs_stepsize": 0.001,
			"random_start": False,
			"steps": 10
		},
		"random_restarts": 5,
	},

	3: {
		"attack": "CFoolboxPGDLinf",
		"model_id": 4,
		"transfer": 3,
		"attack_params": {
			"y_target": None,
			"epsilons": 0.031,
			"abs_stepsize": 0.01,
			"random_start": False,
			"steps": 50
		},
		"random_restarts": 5,
	},
}

ADAPTIVE_ATTACKS = {
	0: {
		"attack": "CFoolboxAveragedPGD",
		"model_id": 0,
		"attack_params": {
			"y_target": None,
			"epsilons": 0.031,
			"abs_stepsize": 0.003,
			"random_start": False,
			"steps": 50,
			"k": 200,
			"sigma": 0.031
		},
		"random_restarts": 5
	},

	1: {
		"attack": "CFoolboxLogitsPGD",
		"model_id": 1,
		"attack_params": {
			"y_target": None,
			"epsilons": 0.3,
			"abs_stepsize": 0.1,
			"random_start": False,
			"steps": 50
		},
		"random_restarts": 5
	},

	2: {
		"attack": "CFoolboxPGDLinf",
		"model_id": 2,
		"attack_params": {
			"y_target": None,
			"epsilons": 0.01,
			"abs_stepsize": 0.01,
			"random_start": False,
			"steps": 50
		},
		"random_restarts": 5,
	},

	3: {
		"attack": "CFoolboxPGDLinfAdaptive",
		"model_id": 3,
		"attack_params": {
			"y_target": None,
			"epsilons": 0.031,
			"abs_stepsize": 0.01,
			"random_start": False,
			"steps": 50
		},
		"random_restarts": 5,
	},
}

AUTO_PGD_ATTACKS = {
	0: {
		"attack": "CAutoAttackAPGDDLR",
		"model_id": 0,
		"attack_params": {
			"dmax": 0.031,
			"y_target": None,
			"epsilons": False,
			"version": "plus"},
	},
	1: {
		"attack": "CAutoAttackAPGDDLR",
		"model_id": 1,
		"attack_params": {
			"dmax": 0.3,
			"y_target": None,
			"epsilons": False,
			"version": "plus"
		},
	},
	2:
		{
			"attack": "CAutoAttackAPGDDLR",
			"model_id": 2,
			"attack_params": {
				"dmax": 0.031,
				"y_target": None,
				"epsilons": False,
				"version": "plus",
			}
		},
	3: {
		"attack": "CAutoAttackAPGDDLR",
		"model_id": 4,
		"transfer": 3,
		"attack_params": {
			"dmax": 0.031,
			"y_target": None,
			"epsilons": False,
			"version": "plus"
		},
	}
}

ATTACK_CLASSES = {
	'CFoolboxPGDLinf': CFoolboxPGDLinf,
	'CFoolboxAveragedPGD': CFoolboxAveragedPGD,
	'CFoolboxLogitsPGD': CFoolboxLogitsPGD,
	'CFoolboxPGDLinfAdaptive': CFoolboxPGDLinfAdaptive,
	'CAutoAttackAPGDCE': CAutoAttackAPGDCE,
	'CAutoAttackAPGDDLR': CAutoAttackAPGDDLR,
	'CAutoAttackAPGDDLRAdaptive': CAutoAttackAPGDDLRAdaptive,
}


def _load_attack(model_id, attack_dict):
	check_model_id(model_id)
	attack_cls = ATTACK_CLASSES[attack_dict[model_id]['attack']]
	model = load_model(attack_dict[model_id]['model_id'])
	attack_args = attack_dict[model_id]['attack_params']
	attack = attack_cls(model, **attack_args)
	if "transfer" in attack_dict[model_id]:
		model_transfer = load_model(attack_dict[model_id]['transfer'])
	else:
		model_transfer = model
	if "random_restarts" in attack_dict[model_id]:
		n_restarts = attack_dict[model_id]['random_restarts']
	else:
		n_restarts = 0
	return attack, model, model_transfer, n_restarts


def load_attack(model_id: int):
	return _load_attack(model_id, ORIGINAL_ATTACKS)


def load_mitigated_attack(model_id: int):
	return _load_attack(model_id, ADAPTIVE_ATTACKS)


def load_auto_attack(model_id: int):
	return _load_attack(model_id, AUTO_PGD_ATTACKS)
