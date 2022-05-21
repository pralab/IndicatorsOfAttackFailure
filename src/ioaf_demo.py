import argparse
import logging
import os

import pandas as pd

from src.attacks.attack_loader import load_attack, load_mitigated_attack
from src.indicators.indicators import compute_indicators
from src.models.data_loader import load_data
from src.models.model_loader import MODELS

MODEL_NAMES = MODELS.keys()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='ID of the model to use. '
                             'Available models are: ' + \
                             " ".join(
                                 [f"{mname}({i})"
                                  for i, mname in enumerate(MODEL_NAMES)]),
                        default='distillation', choices=MODEL_NAMES)
    parser.add_argument('--samples', type=int,
                        help='Number of samples to use.', default=10)
    args = parser.parse_args()
    model_id = args.model
    if not model_id in MODEL_NAMES:
        raise ValueError("Model ID not in list of available models. "
                         "Available models are:\n" + "\n".join(MODEL_NAMES))
    N_SAMPLES = args.samples

    # create logger
    logger = logging.getLogger('progress')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('progress.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Evaluating model {model_id}...")

    X, Y = load_data(model_id, n_samples=N_SAMPLES)
    attack_orig, model_orig, transfer_model_orig, n_restarts_orig = \
        load_attack(model_id)  # original eval
    attack_mitig, model_mitig, transfer_model_mitig, n_restarts_mitig = \
        load_mitigated_attack(model_id)  # mitigation

    all_indicators_orig_eval = []
    all_indicators_patched_eval = []
    for sample in range(N_SAMPLES):
        x, y = X[sample, :], Y[sample]
        pred = model_orig.predict(x)  # model under attack

        logger.info(f"Point {sample + 1}/{N_SAMPLES}")

        logger.info("Computing indicators for original eval")
        indicators_orig = compute_indicators(attack_orig, x, y,
                                             model_orig,
                                             transfer_model_orig,
                                             n_restarts_orig)
        all_indicators_orig_eval.append(indicators_orig)

        logger.info("Computing indicators for patched eval")
        indicators_mitig = compute_indicators(attack_mitig, x, y,
                                              model_mitig,
                                              transfer_model_mitig,
                                              is_patched=True)
        all_indicators_patched_eval.append(indicators_mitig)

    all_indicators_orig_eval = pd.concat(all_indicators_orig_eval,
                                         axis=0)
    all_indicators_patched_eval = pd.concat(all_indicators_patched_eval,
                                            axis=0)

    logger.info("Evaluation complete. Storing indicators in csv report")
    base_path = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    all_indicators_orig_eval.to_csv(os.path.join(base_path, f'{model_id}_indicators_original_eval.csv'))
    all_indicators_patched_eval.to_csv(os.path.join(base_path, f'{model_id}_indicators_mitigation_eval.csv'))

    logger.debug("Logging all the results")
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        logger.debug(all_indicators_orig_eval)
        logger.debug(all_indicators_patched_eval)
