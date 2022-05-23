import os

import pandas as pd

results_dir = 'src/results'

indicators_data = []
for f in os.listdir(results_dir):
    fname = os.path.join(results_dir, f)
    data = pd.read_csv(fname)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(fname)
        print(data)
    indicators_data.append(data)

indicators_data = pd.concat(indicators_data)

