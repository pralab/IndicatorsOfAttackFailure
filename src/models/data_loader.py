import numpy as np
from secml.data import CDataset
from secml.data.loader import CDataLoaderMNIST
from secml.data.loader.c_dataloader_cifar import CDataLoaderCIFAR, CDataLoaderCIFAR10

from src.models.ensemble_diversity.load_ensemble import reshape_cifar10
from src.models.model_loader import check_model_id


def load_data(model_id: str, n_samples: int = 100):
	check_model_id(model_id)

	if model_id == 'distillation':
		return _load_mnist_for_distillation(n_samples)
	else:
		return _load_cifar_regular(n_samples)

def _load_mnist_for_distillation(n_samples: int):
	ts = CDataLoaderMNIST().load('testing')
	pt = random_sample(ts, n_samples)
	x0, y0 = pt.X / 255., pt.Y
	return x0, y0


def _load_cifar_regular(n_samples: int):
	_, ts = CDataLoaderCIFAR10().load()
	pt = random_sample(ts, n_samples)
	x0, y0 = pt.X / 255., pt.Y
	return x0, y0


def _load_cifar_ensemble(n_samples: int):
	# needed for tf models
	_, ts = CDataLoaderCIFAR10().load()
	ts = random_sample(ts, n_samples)
	reshaped_pts = reshape_cifar10(ts)
	normalized_pts = reshaped_pts.X / 255.
	return normalized_pts, ts.Y

def random_sample(ds, n_samples):
	ds_size = ds.X.shape[0]
	indexes = np.random.choice(range(ds_size), size=n_samples, replace=False)
	return ds[indexes.tolist(), :]