## This file concatenates the results across all the models for the same dataset, takes the median of errors for different seeds and writes the mean SMAPE, median SMAPE, ranked SMAPE,
# mean MASE, median MASE and ranked MASE to a csv file

import glob
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

# get the different cluster names as external arguments
argument_parser = argparse.ArgumentParser("Ensembling forecasts")
argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')

# parse the user arguments
args = argument_parser.parse_args()
dataset_name = args.dataset_name

input_path = '../results/rnn_forecasts/'

output_path = '../results/ensemble_rnn_forecasts/'

all_forecast_files = [filename for filename in glob.iglob(input_path + dataset_name + "_*")]

all_models_all_seeds_forecasts_dic = defaultdict(list)

# concat all the errors to data frames
for file_name in sorted(all_forecast_files):
    file_name_part = file_name.rsplit('_', 1)[0]
    file_name_part = file_name_part.split(input_path, 1)[1]

    # read the forecasts from the current file
    current_model_forecasts_df = pd.read_csv(file_name, header=None, dtype=np.float64)

    # append the forecasts to the dictionary
    all_models_all_seeds_forecasts_dic[file_name_part].append(current_model_forecasts_df)

# iterate the dictionary
for (model, forecasts_df_list) in all_models_all_seeds_forecasts_dic.items():

    # convert the dataframe list to an array
    forecasts_array = np.stack(forecasts_df_list)

    # take the mean of forecasts across seeds for ensembling
    ensembled_forecasts = np.nanmedian(forecasts_array, axis=0)

    # write the ensembled forecasts to a file
    output_file = output_path + model
    np.savetxt(output_file, ensembled_forecasts, delimiter = ',')



