import os

import robustbench
from secml.array import CArray
from secml.data.loader import CDataLoaderCIFAR10

from src.models.tws.tws_wrapper import CClassifierPytorchYuHu


def _load_model(hide_reject=False):
	model = robustbench.utils.load_model(
		'Standard', norm='Linf', model_dir=os.path.join(
			os.path.dirname(__file__), '..', '..', '..', 'models'))
	model.eval()
	clf = CClassifierPytorchYuHu(model=model, input_shape=(3, 32, 32),
								 threshold=1000, hide_reject=False)
	_, cifar_ts = CDataLoaderCIFAR10().load()
	idxs = CArray.arange(0, cifar_ts.num_samples)
	idxs.shuffle()
	idxs = idxs[:1000]
	cifar_ts = cifar_ts[idxs, :]
	cifar_ts.X /= 255.
	clf.threshold = 1.9999676942825317
	# uncomment for recomputing the reject threshold
	# clf.threshold = clf.compute_threshold(0.2, cifar_ts)
	clf.hide_reject = hide_reject
	return clf


def load_model():
	return _load_model(hide_reject=False)


def load_model_no_reject():
	return _load_model(hide_reject=True)
