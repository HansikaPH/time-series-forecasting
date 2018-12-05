## This file calculates the ranked SMAPE and ranked MASE of all models for a given dataset and a seed

import glob
import argparse
import pandas as pd
import re
import numpy as np

# get the different cluster names as external arguments
argument_parser = argparse.ArgumentParser("Calculate ranked errors")
argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
argument_parser.add_argument('--is_merged_cluster_result', required=True, help='1/0 denoting whether the results are merged from multiple clusters or not')
argument_parser.add_argument('--model_seed_number', required=True, help='In case there are multiple results of the same model with different seeds')

# parse the user arguments
args = argument_parser.parse_args()
dataset_name = args.dataset_name
is_merged_cluster_result = args.is_merged_cluster_result
seed = int(args.model_seed_number)


if int(is_merged_cluster_result):
    input_path = '../results/errors/merged_cluster_results/'
else:
    input_path = '../results/errors/'

output_path = input_path

output_file = output_path + "ranked_mean_" + dataset_name + "_"

# get the list of all the files matching the regex
all_SMAPE_files = [filename for filename in glob.iglob(input_path + "all_smape_errors_" + dataset_name + "_*")]
all_MASE_files = [filename for filename in glob.iglob(input_path + "all_mase_errors_" + dataset_name + "_*")]

all_smape_errors_df = pd.DataFrame()
all_mase_errors_df = pd.DataFrame()

# concat all the errors to data frames
for smape_errors_file, mase_errors_file in zip(sorted(all_SMAPE_files),
                                                         sorted(all_MASE_files)):
    smape_errors_file_object = open(smape_errors_file, "r")
    mase_errors_file_object = open(mase_errors_file, "r")

    file_name_part = re.split(pattern="all_smape_errors_" + dataset_name + "_" , string=smape_errors_file, maxsplit=1)[1]

    try:
        model_seed = int(file_name_part.rsplit('_', 1)[1].split('.')[0])
        if model_seed != seed:
            continue
    except:
        pass

    # calculate the ranked smape
    current_model_all_smape_errors = [float(num) for num in smape_errors_file_object]
    all_smape_errors_df[file_name_part] = current_model_all_smape_errors

    # calculate the ranked mase
    current_model_all_mase_errors = [float(num) for num in mase_errors_file_object]
    all_mase_errors_df[file_name_part] = current_model_all_mase_errors

# calculate the ranked errors
all_smape_ranks_df = all_smape_errors_df.rank(axis=1)
all_mase_ranks_df = all_mase_errors_df.rank(axis=1)

smape_ranked_errors = np.mean(all_smape_ranks_df, axis=0)
mase_ranked_errors = np.mean(all_mase_ranks_df, axis=0)

for ranked_smape, ranked_mase, model_name in zip(smape_ranked_errors, mase_ranked_errors, smape_ranked_errors.index.values):
    output_file_name = output_file + model_name

    # write the results to file
    file_object = open(output_file_name, "w")
    file_object.writelines("ranked_SMAPE:" + str(ranked_smape) + "\n")
    file_object.writelines("ranked_MASE:" + str(ranked_mase))
