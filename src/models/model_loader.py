from src.models.advt import load_advt
from src.models.das import load_das
from src.models.distillation import load_distillation
from src.models.externals.dnr import load_dnr
from src.models.ensemble_diversity import load_ensemble
from src.models.guo import load_guo
from src.models.kwta import load_kwta
from src.models.standard import load_standard
from src.models.tws import load_tws

MODELS = {
    'kwta': load_kwta.load_model,
    'distillation': load_distillation.load_model,
    'pang': load_ensemble.load_model,
    'tws': load_tws.load_model,
    'tws_no_reject': load_tws.load_model_no_reject,
    'dnr': load_dnr.load_model,
    'das': load_das.load_model,
    'das_undef': load_das.load_undefended_model,
    'das_bpda': load_das.load_bpda_model,
    'guo': load_guo.load_model,
    'standard': load_standard.load_model,
    'advt': load_advt.load_model,
}


def load_model(model_id: str):
    check_model_id(model_id)
    model = MODELS[model_id]()
    return model


def check_model_id(model_id):
    if model_id not in MODELS:
        raise ValueError(f'{model_id} not in the list of available models.')
