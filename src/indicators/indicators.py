import numpy as np
import pandas as pd
import torch
from secml.adv.attacks import CAttackEvasion, CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor, as_carray
from secml.array import CArray
from secml.core.constants import inf
from secml.ml import CClassifier
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR

from src.attacks.smoothed_pgd import CFoolboxAveragedPGD

REJECT_CLASSES = [-1, 10]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _extract_input_shape(model: CClassifier):
    if isinstance(model, CClassifierDNR):
        return (1, 3, 32, 32)
    elif isinstance(model, CClassifierRejectThreshold):
        return (1, *model._clf.input_shape)
    else:
        return (1, *model.input_shape)


def interrupted_computational_graph_metric(model: CClassifier) -> bool:
    """
    Checks if the model is end-to-end differentiable.
    :param model: model to test
    :return: interrupted graph metric
    """
    input_shape = (1, *_extract_input_shape(model))
    random_sample = CArray.rand(input_shape)
    y = model.decision_function(random_sample)
    try:
        model.gradient(random_sample, y)
    except Exception:
        return True
    return False


def zero_gradients_metric(attack: CAttackEvasion, x: CArray) -> bool:
    """
    Counts if any of the samples has zero gradients w.r.t. the attack loss.
    :param attack: attack to be used for the evaluation
    :param x: batch of samples to use for the test
    :return: zero gradients metric
    """
    preds, scores = attack.classifier.predict(x, return_decision_function=True)
    n_zero_grads = 0
    n_samples = preds.shape[0]
    for i in range(n_samples):
        # this is a fix required for CAttackEvasionFoolbox
        attack._y0 = as_tensor(preds[i]) if isinstance(attack, CAttackEvasionFoolbox) else preds[i]
        grad_norm = attack.objective_function_gradient(x[i, :]).norm(0)
        n_zero_grads += 1 if grad_norm == 0 else 0
    return n_zero_grads > 0


def unavailable_gradients_indicator(model: CClassifier, attack: CAttackEvasion, x: CArray) -> bool:
    """
    Computes if the model has valid and usable gradients.
    :param model: model to use for the evaluation
    :param attack: attack to use for the evaluation
    :param x: batch of input samples
    :return: the unavailable gradients indicator
    """
    interrupted_graph = interrupted_computational_graph_metric(model)
    zero_grads = zero_gradients_metric(attack, x)
    return interrupted_graph or zero_grads


# TODO this can be implemented with fewer conversions between CArrays, ndarrays and tensors
def unstable_predictions_indicator(attack: CAttackEvasion, x: CArray, y: CArray, gamma: int = 100,
                                   radius: float = 8 / 255) -> float:
    attack._y0 = as_tensor(y) if isinstance(attack, CAttackEvasionFoolbox) else y

    # we need to switch off the detector
    clf_backup = attack.classifier
    if isinstance(attack.classifier, CClassifierRejectThreshold):
        # The rejection is not part of the prediction, hence it is not considered by this indicator
        attack._classifier = attack.classifier.clf
    if isinstance(attack, CFoolboxAveragedPGD):
        xt = as_tensor(x).repeat_interleave(attack.k, dim=0)
        attack._y0 = attack._y0.repeat_interleave(attack.k, dim=0)
        noise = (torch.rand(xt.shape, device=device) - 0.5) * attack.sigma
        reference_loss = as_tensor(attack.objective_function(as_carray(xt + noise)))
        reference_loss = as_carray(reference_loss.view(attack.k, x.shape[0]).mean(dim=0))
    else:
        reference_loss = attack.objective_function(x)
    relative_increments = CArray.zeros((gamma, x.shape[0]))
    divide_reference = CArray(reference_loss.tondarray()).abs()
    divide_reference[divide_reference == 0] = 1
    for i in range(gamma):
        random_directions = (torch.rand(x.shape, device=device) - 0.5) * radius
        x_perturbed = as_tensor(x) + random_directions
        if isinstance(attack, CFoolboxAveragedPGD):
            x_sigma = x_perturbed.repeat_interleave(attack.k, dim=0)
            noise = (torch.rand(x_sigma.shape, device=device) - 0.5) * attack.sigma
            avg_loss = as_tensor(attack.objective_function(as_carray(x_sigma + noise)))
            avg_loss = avg_loss.view(attack.k, x.shape[0])
            losses = as_carray(avg_loss.mean(dim=0))
        else:
            losses = attack.objective_function(as_carray(x_perturbed))
        relative_increment = CArray.abs(reference_loss - losses)
        relative_increment /= divide_reference
        relative_increments[i, :] = relative_increment
    metric = relative_increments.mean()
    metric = min(metric, 1.0)

    # set again the original clf
    attack._classifier = clf_backup
    return metric


def silent_success_indicator(attack: CAttackEvasion, y0: int, adv_y: int) -> bool:
    """
    Computes the presence of adversarial examples in the path.
    :param attack: the attack to use for the evaluation
    :param y0: the real label
    :param adv_y: the label after the attack
    :return: the silent success indicator
    """
    scores = attack.classifier.predict(attack.x_seq)
    y_target = attack.y_target
    if y_target is None:
        scores_success = scores.argmax(axis=1) != y0
    else:
        scores_success = scores.argmax(axis=1) == y_target
    attack_success = adv_y != y0 if y_target is None else adv_y == y_target
    return not attack_success and scores_success[:-1].sum() != 0


def always_decreasing(x):
    """
    Makes sure a sequence of numbers is always decreasing.
    :param x: input sequence
    :return: the transformed monotonically-decreasing sequence
    """
    minimum = x[0]
    for i in range(x.shape[0]):
        if x[i] < minimum:
            minimum = x[i]
        else:
            x[i] = minimum
    return x


def rescale_loss(atk_loss):
    """
    Rescales the loss in the [0, 1] interval, handling the case
    where the loss is always zero.
    :param atk_loss: loss to be rescaled
    :return: the rescaled loss
    """
    if atk_loss.max() != 0:
        atk_loss -= atk_loss.min()
        atk_loss /= atk_loss.max()
    atk_loss = atk_loss.ravel()
    return atk_loss


def incomplete_optimization_indicator(loss):
    """
    Checks if the optimization has converged to a minimum
    :param loss: the loss computed by the attack along the iterations
    :type loss: np.ndarray
    :return: the incomplete optimization metric
    :rtype: float
    """
    loss = rescale_loss(loss)
    loss = always_decreasing(loss)
    num_steps = min(len(loss), 10)
    last_steps = loss[-num_steps:]
    diffs = last_steps.max() - last_steps.min()
    return float(diffs > 0.05)


def transfer_failure_indicator(transfer_scores: np.ndarray, pred: int) -> bool:
    """
    Computes how many attacks (optimized against the surrogate) fail against the real model.
    :param transfer_scores: the scores computed on the real model
    :param pred: the label predicted by the surrogate
    :return: the non-transferability indicator
    """
    return pred != transfer_scores


def unconstrained_attack_failure_indicator(x, y, attack: CAttackEvasion) -> float:
    """
    Checks for how many points (among the input points passed)
    the attack with no bounds reaches the adversarial region.
    :param x: input samples
    :param y: label of the input samples
    :param attack: attack to use for the evaluation
    :return: the unconstrained attack failure indicator
    """
    attack.dmax = inf
    failed = torch.zeros(x.shape[0])

    # set unconstrained bound
    if hasattr(attack, 'epsilons'):
        attack.epsilons = np.inf
    if hasattr(attack, 'epsilon'):
        attack.epsilon = np.inf

    for i in range(x.shape[0]):
        x_i, y_i = x[i, :], y[i]
        y_pred = attack.classifier.predict(x_i)
        if y_pred != y_i:
            failed[i] = 0.0
            continue
        adv_label, scores, _, _ = attack.run(x_i, y_i)
        silent = silent_success_indicator(attack, y0=y_i.item(), adv_y=adv_label)
        failed[i] = float(not silent and (adv_label == y_pred or adv_label == -1).item())
    sanity_check_metric = failed.mean().item()
    return sanity_check_metric


def attack_fails(adv_pred, target_label, y0, transfer_scores=None) -> bool:
    """
    Checks if the attack fails for the given objective.
    :param adv_pred: the predicted label of the perturbed sample
    :param target_label: the target label of the attack (None if untargeted)
    :param y0: the original label of the unperturbed sample
    :param transfer_scores: optional predicted label of the perturbed sample
        when passed to a target model (transfer attack)
    :return: boolean value, true if the attack succeeded
    """
    untarget_fail = target_label is None and adv_pred == y0
    targeted_fail = target_label is not None and adv_pred != target_label
    rejection_failure = adv_pred == -1 or (transfer_scores is not None and transfer_scores in REJECT_CLASSES)
    return untarget_fail or targeted_fail or rejection_failure


def compute_indicators(attack, x, y, clf, transfer_clf, is_patched=False) -> pd.DataFrame:
    """
    Computes all the indicators of attack failure.
    :param attack: the attack used for the evaluation
    :param x: the input samples
    :param y: the ground-truth labels of the input samples
    :param clf: classifier to use for crafting the attack
    :param transfer_clf: classifier to use for testing the attack results
    :param is_patched: boolean value to be set for patched evaluations (checks
        for silent success and transfer failures)
    :return: a dataframe with all the values of the indicators
    """
    # run attack
    y_adv, _, adv_ds, f_opt = attack.run(x, y)

    # get information required for computing the indicators
    y_real = CArray(attack._y0)
    transfer_scores = transfer_clf.predict(adv_ds.X)
    y_target = attack.y_target
    attacker_loss = attack.objective_function(attack.x_seq).tondarray()
    attack_failed = attack_fails(y_adv, y_target, y_real, transfer_scores)

    # compute indicators
    unavailable_gradients = unavailable_gradients_indicator(clf, attack, x)
    unstable_predictions = unstable_predictions_indicator(attack, x, y)
    silent_success = silent_success_indicator(attack, y, y_adv)  # can be switched to targeted as well
    incomplete_optimization = incomplete_optimization_indicator(attacker_loss)
    transfer_failure = transfer_failure_indicator(transfer_scores, y_adv) \
        if transfer_scores is not None else False
    unconstrained_attack_failure = unconstrained_attack_failure_indicator(x, y, attack)
    if is_patched:
        attack_failed = attack_failed \
                        and not silent_success \
                        and not transfer_failure
    else:
        attack_failed = attack_failed and not transfer_failure

    df = pd.DataFrame(data={
        'attack_success': 0.0 if attack_failed else 1.0,
        'unavailable_gradients': unavailable_gradients,
        'unstable_predictions': unstable_predictions,
        'silent_success': silent_success,
        'incomplete_optimization': incomplete_optimization,
        'transfer_failure': transfer_failure,
        'unconstrained_attack_failure': unconstrained_attack_failure,
    }, index=[0])

    return df
