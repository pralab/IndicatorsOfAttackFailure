import numpy as np
import pandas as pd
import sklearn
from secml.array import CArray

from src.utils.c_metric_score_difference import CMetricScoreDifference
from src.utils.sampling import sampling_n_sphere

REJECT_CLASSES = [-1, 10]


def transfer_failure_indicator(transfer_scores: np.ndarray, pred: int):
	"""
	Computes how many attacks (optimized against the surrogate) fail against the real model.
	:param transfer_scores: the scores computed on the real model
	:type transfer_scores: np.ndarray
	:param pred: the label predicted by the surrogate
	:type pred: int
	:return: the non-transferability indicator
	:rtype: bool
	"""
	return pred != transfer_scores


def break_point_angle_indicator(loss: np.ndarray):
	"""
	Computes the break-point angle, that quantifies the failure due to small hyperparameters
	:param loss: the loss computed by the attack, along the attack path
	:type loss: np.ndarray
	:return: the break-point indicator
	:rtype: float
	"""
	a, c = np.array([0, loss[0]]), np.array([1, loss[-1]])
	v1 = c - a
	v1 /= np.linalg.norm(v1)
	distances = []
	for i in range(1, len(loss)):
		p = np.array([i * 1 / len(loss), loss[i]]) - a
		distances.append(np.linalg.norm(p - p.T.dot(v1) * v1))
	best_index = np.argmax(distances) + 1
	b = np.array([1 / len(loss) * best_index, loss[best_index]])
	v2 = b - a
	v2 /= np.linalg.norm(v2)
	v3 = b - c
	v3 /= np.linalg.norm(v3)
	alignment = v2.T.dot(v3)
	alignment = np.abs(alignment)
	return alignment


def silent_success_indicator(loss: np.array, scores: np.ndarray, y0: int, y_target: int, adv_y: int):
	"""
	Computes the presence of adversarial examples in the path
	:param loss: the loss computed by the attack, along the attack path
	:type loss: np.ndarray
	:param scores: the scores of the attack path, computed against the target
	:type scores: np.ndarray
	:param y0: the real label
	:type y0: int
	:param y_target: the target label of the attack. None for untarget
	:type y_target: int
	:param adv_y: the label after the attack
	:type adv_y: int
	:return: the silent success indicator
	:rtype: bool
	"""
	if y_target is None:
		scores_success = np.argmax(scores, axis=1) != y0
	else:
		scores_success = np.argmax(scores, axis=1) == y_target
	zero_index = np.argmax(scores_success)
	adv_ratio = (zero_index / loss.shape[0]) if zero_index > 0 else 1
	attack_success = adv_y != y0 if y_target is None else adv_y == y_target
	return adv_ratio != 1 and not attack_success


def increasing_loss_indicator(atk_loss):
	"""
	Compute the presence of noisy gradients, by estimating how much the loss is increasing during the optimization
	:param atk_loss: the loss computed by the attack, along the attack path
	:type atk_loss: np.ndarray
	:return: the increasing loss indicator
	:rtype: float
	"""
	increasing = []
	for i in range(1, atk_loss.shape[0]):
		diff = atk_loss[i].item() - atk_loss[i - 1].item()
		v = np.maximum(0.0, diff)
		if v != 0:
			increasing.append(atk_loss[i].item())
			increasing.append(atk_loss[i - 1].item())
	if len(increasing) < 2:
		return 0, 0
	auc = sklearn.metrics.auc(np.linspace(0, 1, len(increasing)), np.array(increasing))
	st = np.std(increasing)
	return auc, st


def bad_init_indicator(y_real, y_pred, y_adv, y_target=None, rejected_class=None):
	"""
	Compute if there are successful attacks in the restarts
	:param y_real: the real label
	:type y_real: int
	:param y_pred: the predicted label after the attack
	:type y_pred: int
	:param y_adv: the labels obtained through restarts
	:type y_adv: list
	:param y_target: the target label of the attack. None for untargeted
	:type y_target: int
	:param rejected_class: list of labels that correspond to rejection classes
	:type rejected_class: list
	:return: true if at least one restart was successful
	:rtype: bool
	"""
	if rejected_class is None:
		rejected_class = REJECT_CLASSES
	if y_target is None:
		metric = (y_pred == y_real or y_pred in rejected_class) \
				 and any([y_real != y and y not in rejected_class for y in y_adv])
	else:
		metric = y_pred != y_target and any([y_target == y for y in y_adv])
	return metric


def zero_gradients_indicator(grad_norm: np.ndarray, zero_index: int, threshold: float = 0):
	"""
	Compute how many of the gradients have zero norm along the path (before finding an adversarial point)
	:param grad_norm: the norms of the gradient
	:type grad_norm: np.ndarray
	:param zero_index: the index of the first adversarial point along the path
	:type zero_index: int
	:param threshold: the threshold on the norms of gradients
	:type threshold: float
	:return: the zero gradients indicator
	:rtype: float
	"""
	how_many = zero_index if zero_index >= 0 else len(grad_norm)
	return (grad_norm[:how_many] <= threshold).mean()


def compute_indicators(attack, x, y, clf, transfer_clf,
					   n_restarts=None, is_patched=False):
	# run attack original eval
	y_adv, scores, adv_ds, f_opt = attack.run(x, y)

	y_real = CArray(attack._y0)

	scores_path = clf.decision_function(attack.x_seq).tondarray()
	transfer_scores = transfer_clf.predict(adv_ds.X)

	y_target = attack.y_target

	grad_path = CArray.zeros(attack.x_seq.shape)
	for i in range(attack.x_seq.shape[0]):
		grad_path[i, :] = attack.objective_function_gradient(attack.x_seq[i, :])
	grad_norms = grad_path.norm_2d(axis=1).tondarray()

	attacker_loss = attack.objective_function(attack.x_seq).tondarray()

	atk_loss_normalized = rescale_loss(attacker_loss)
	attack_failed = attack_fails(y_adv, y_target, y_real, transfer_scores)
	_, _, diff = CMetricScoreDifference.score_difference(scores_path,
														 y_real.tondarray(),
														 y_target)
	zero_index = (diff < 0).argmax()
	if zero_index == 0 and diff[0] >= 0:
		zero_index = -1
	increasing_loss, std_loss = increasing_loss_indicator(atk_loss_normalized)
	break_point_angle = break_point_angle_indicator(atk_loss_normalized)
	silent_success = silent_success_indicator(attacker_loss, scores_path,
											  y_real.item(), y_target, y_adv)

	if n_restarts is not None:
		restart_label_results = []
		for j in range(n_restarts):
			perturb_param = attack.epsilon
			random_restarted_sample = sampling_n_sphere(x.tondarray(),
														perturb_param, p=np.inf)
			y_pred_r, scores_r, adv_ds_r, f_obj_r = attack.run(random_restarted_sample, y)
			if transfer_clf is not None:
				# if a transfer clf is defined, the restarts
				# should be evaluated in the transfer clf
				y_pred_r = transfer_clf.predict(adv_ds_r.X)
			else:
				y_pred_r = clf.predict(adv_ds_r.X)
			restart_label_results.append(y_pred_r)

	bad_init_value = bad_init_indicator(y_real=y_real, y_pred=y_adv,
										y_adv=restart_label_results,
										y_target=y_target) if n_restarts is not None else False
	zero_grads = zero_gradients_indicator(grad_norms, zero_index)
	transfer_failure = transfer_failure_indicator(transfer_scores, y_adv) \
		if transfer_scores is not None else False
	if is_patched:
		attack_failed = attack_failed \
						and not bad_init_value \
						and not silent_success \
						and not transfer_failure
	else:
		attack_failed = attack_failed and not transfer_failure

	df = pd.DataFrame(data={
		'Attack Success': 0.0 if attack_failed else 1.0,
		'Silent Success': 1.0 if silent_success else 0.0,
		'Break-point Angle': break_point_angle,
		'Increasing Loss': increasing_loss,
		'Zero Gradients': zero_grads,
		'Bad Initialization': bad_init_value,
		'Transfer Failure': transfer_failure,
	}, index=[0])

	return df


def rescale_loss(atk_loss):
	if atk_loss.max() != 0:
		atk_loss -= atk_loss.min()
		atk_loss /= atk_loss.max()
	atk_loss = atk_loss.ravel()
	return atk_loss


def attack_fails(adv_pred, target_label, y0, transfer_scores):
	untarget_fail = target_label is None and adv_pred == y0
	targeted_fail = target_label is not None and adv_pred != target_label
	rejection_failure = adv_pred == -1 or (transfer_scores is not None and transfer_scores in REJECT_CLASSES)
	return untarget_fail or targeted_fail or rejection_failure
