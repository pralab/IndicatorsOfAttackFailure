import numpy as np


def sampling_n_sphere(x, eps: float, p=np.inf):
	c = np.random.uniform(low=-1, high=1, size=x.shape).ravel()
	c = c / np.linalg.norm(c, ord=p) * eps
	return x + c
