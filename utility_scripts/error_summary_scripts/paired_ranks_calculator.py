import pandas as pd
import numpy as np

input_file = "../../results/statistical_significance_tests/stl_comparison_nn5.csv"

data = pd.read_csv(input_file)

ranks = np.mean(data.rank(axis=1, ascending = False), axis=0)

print(ranks)