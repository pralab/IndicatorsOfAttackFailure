# Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples

Preprint available at [https://arxiv.org/abs/2106.09947](https://arxiv.org/abs/2106.09947).

The code is developed with [SecML](https://secml.gitlab.io/) library.

For computing the indicators, run the following command:

```bash
python3 -m src.ioaf_demo --model <model_id> --samples <num_samples>
```

For a complete list of models, run: 

```bash
python3 -m src.ioaf_demo --help
```

For a complete example, run `run_all.sh`.



## Models used in the paper

* k-WTA: [https://github.com/a554b554/kWTA-Activation](https://github.com/a554b554/kWTA-Activation)
  
* Distillation: adapted from the code found at [https://github.com/carlini/nn_robust_attacks](https://github.com/carlini/nn_robust_attacks)

* Ensemble Diversity: [https://github.com/P2333/Adaptive-Diversity-Promoting](https://github.com/P2333/Adaptive-Diversity-Promoting)

* TWS: adapted from [https://github.com/s-huu/TurningWeaknessIntoStrength](https://github.com/s-huu/TurningWeaknessIntoStrength)

## Other sources

* Adaptive attacks: [https://github.com/wielandbrendel/adaptive_attacks_paper](https://github.com/wielandbrendel/adaptive_attacks_paper)
