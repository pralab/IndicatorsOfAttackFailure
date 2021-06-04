from src.models.distillation import load_distillation
from src.models.ensemble_diversity import load_ensemble
from src.models.kwta import load_kwta
from src.models.tws import load_tws

MODELS = {
	0: load_kwta.load_model,
	1: load_distillation.load_model,
	2: load_ensemble.load_model,
	3: load_tws.load_model,
	4: load_tws.load_model_no_reject
}


def load_model(model_id: int):
	check_model_id(model_id)
	model = MODELS[model_id]()
	return model


def check_model_id(model_id):
	if model_id < 0 or model_id > len(MODELS):
		raise ValueError(f'{model_id} not in the range of provided model. Use 0,1,2,3 or 4')
