from secml.adv.attacks import CFoolboxPGDLinf, CFoolboxDeepfool

from src.attacks.autoattack_wrapper import CAutoAttackAPGDCE, CAutoAttackAPGDDLR
from src.attacks.pgd_adaptive import CFoolboxPGDLinfAdaptive
from src.attacks.pgd_logits import CFoolboxLogitsPGD
from src.attacks.smoothed_pgd import CFoolboxAveragedPGD
from src.attacks.unnormalized_pgd import CPGDLInfUnnormalized
from src.models.model_loader import check_model_id, load_model

ORIGINAL_ATTACKS = {
    'kwta': {
        "attack": "CFoolboxPGDLinf",
        "model_id": 'kwta',
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.01,
            "random_start": False,
            "steps": 50
        },
    },

    'distillation': {
        "attack": "CFoolboxPGDLinf",
        "model_id": 'distillation',
        "attack_params": {
            "y_target": None,
            "epsilons": 0.3,
            "abs_stepsize": 0.1,
            "random_start": False,
            "steps": 50
        },
    },

    'pang': {
        "attack": "CFoolboxPGDLinf",
        "model_id": 'pang',
        "attack_params": {
            "y_target": None,
            "epsilons": 0.01,
            "abs_stepsize": 0.001,
            "random_start": False,
            "steps": 10
        },
    },

    'tws': {
        "attack": "CFoolboxPGDLinf",
        "model_id": "tws_no_reject",
        "transfer": "tws",
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.01,
            "random_start": False,
            "steps": 50
        },
    },

    'das': {
        "attack": "CFoolboxDeepfool",
        "transfer": "das",
        "model_id": "das_undef",
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "steps": 100
        },
    },

    'guo': {
        "attack": "CFoolboxPGDLinf",
        "model_id": "guo",
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.003,
            "random_start": False,
            "steps": 10
        },
    },

    'dnr': {
        "attack": "CPGDLInfUnnormalized",
        "model_id": "dnr",
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.03,
            "random_start": False,
            "steps": 50
        },
    },
}
ADAPTIVE_ATTACKS = {
    'kwta': {
        "attack": "CFoolboxAveragedPGD",
        "model_id": 'kwta',
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.003,
            "random_start": False,
            "steps": 50,
            "k": 50,
            "sigma": 0.031
        },
    },

    'distillation': {
        "attack": "CFoolboxLogitsPGD",
        "model_id": 'distillation',
        "attack_params": {
            "y_target": None,
            "epsilons": 0.3,
            "abs_stepsize": 0.1,
            "random_start": False,
            "steps": 50
        },
    },

    'pang': {
        "attack": "CFoolboxPGDLinf",
        "model_id": 'pang',
        "attack_params": {
            "y_target": None,
            "epsilons": 0.01,
            "abs_stepsize": 0.01,
            "random_start": False,
            "steps": 50
        },
    },

    'tws': {
        "attack": "CFoolboxPGDLinfAdaptive",
        "model_id": "tws",
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.01,
            "random_start": False,
            "steps": 50
        },
    },

    'das': {
        "attack": "CFoolboxPGDLinf",
        "model_id": "das_bpda",
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.003,
            "steps": 300
        },
    },

    'guo': {
        "attack": "CFoolboxAveragedPGD",
        "model_id": "guo",
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.003,
            "random_start": False,
            "steps": 200,
            "k": 200,
            "sigma": True
        },
    },

    'dnr': {
        "attack": "CFoolboxPGDLinfAdaptive",
        "model_id": "dnr",
        "attack_params": {
            "y_target": None,
            "epsilons": 0.031,
            "abs_stepsize": 0.01,
            "random_start": False,
            "steps": 200
        },
    }
}

AUTO_PGD_ATTACKS = {
    'kwta': {
        "attack": "CAutoAttackAPGDDLR",
        "model_id": 'kwta',
        "attack_params": {
            "dmax": 0.031,
            "y_target": None,
            "epsilons": False,
            "version": "plus"},
    },
    'distillation': {
        "attack": "CAutoAttackAPGDDLR",
        "model_id": 'distillation',
        "attack_params": {
            "dmax": 0.3,
            "y_target": None,
            "epsilons": False,
            "version": "plus"
        },
    },
    'pang': {
        "attack": "CAutoAttackAPGDDLR",
        "model_id": 'pang',
        "attack_params": {
            "dmax": 0.031,
            "y_target": None,
            "epsilons": False,
            "version": "plus",
        }
    },
    'tws': {
        "attack": "CAutoAttackAPGDDLR",
        "model_id": 'tws',
        "attack_params": {
            "dmax": 0.031,
            "y_target": None,
            "epsilons": False,
            "version": "plus"
        },
    },
    'das': {
        "attack": "CAutoAttackAPGDDLR",
        "model_id": 'das_undef',
        "transfer": 'das',
        "attack_params": {
            "dmax": 0.031,
            "y_target": None,
            "epsilons": False,
            "version": "plus"
        },
    },
    'guo': {
        "attack": "CAutoAttackAPGDDLR",
        "model_id": 'guo',
        "attack_params": {
            "dmax": 0.031,
            "y_target": None,
            "epsilons": False,
            "version": "plus"
        },
    },
    'dnr': {
        "attack": "CAutoAttackAPGDDLR",
        "model_id": 'dnr',
        "attack_params": {
            "dmax": 0.031,
            "y_target": None,
            "epsilons": False,
            "version": "plus"
        },
    },
}

ATTACK_CLASSES = {
    'CFoolboxPGDLinf': CFoolboxPGDLinf,
    'CFoolboxAveragedPGD': CFoolboxAveragedPGD,
    'CFoolboxLogitsPGD': CFoolboxLogitsPGD,
    'CFoolboxPGDLinfAdaptive': CFoolboxPGDLinfAdaptive,
    'CAutoAttackAPGDCE': CAutoAttackAPGDCE,
    'CAutoAttackAPGDDLR': CAutoAttackAPGDDLR,
    'CPGDLInfUnnormalized': CPGDLInfUnnormalized,
    'CFoolboxDeepfool': CFoolboxDeepfool,
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
