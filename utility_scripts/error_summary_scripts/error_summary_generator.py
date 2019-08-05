## This file concatenates the results across all the models for the same dataset and writes the mean SMAPE, median SMAPE, ranked SMAPE,
# mean MASE, median MASE and ranked MASE across the time series to a csv file

import glob
import argparse
import pandas as pd
import re
import numpy as np
from collections import defaultdict

# get the different cluster names as external arguments
argument_parser = argparse.ArgumentParser("Create error summaries")
argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
argument_parser.add_argument('--is_merged_cluster_result', required=True,
                             help='1/0 denoting whether the results are merged from multiple clusters or not')

# parse the user arguments
args = argument_parser.parse_args()
dataset_name = args.dataset_name
is_merged_cluster_result = args.is_merged_cluster_result

if int(is_merged_cluster_result):
    input_path = '../results/ensemble_errors/merged_cluster_results/'
else:
    input_path = '../results/ensemble_errors/'

output_path = '../results/ensemble_errors/aggregate_errors/'

output_file = output_path + dataset_name + ".csv"

# get the list of all the files matching the regex
all_SMAPE_files = [filename for filename in glob.iglob(input_path + "all_smape_errors_" + dataset_name + "_*")]
all_MASE_files = [filename for filename in glob.iglob(input_path + "all_mase_errors_" + dataset_name + "_*")]

average_smape_errors_df = pd.DataFrame()
average_mase_errors_df = pd.DataFrame()
all_errors_df = pd.DataFrame(columns=["Model_Name", "Mean_SMAPE", "Median_SMAPE", "Mean_MASE", "Median_MASE"])

all_seeds_smape_errors_dic = {}
all_seeds_mase_errors_dic = {}

# concat all the errors to data frames
for smape_errors_file, mase_errors_file in zip(sorted(all_SMAPE_files),
                                               sorted(all_MASE_files)):

    print(smape_errors_file)
    smape_errors_file_object = open(smape_errors_file, "r")
    mase_errors_file_object = open(mase_errors_file, "r")
    file_name_part = re.split(pattern="all_smape_errors_" + dataset_name + "_", string=smape_errors_file, maxsplit=1)[1]


    current_model_all_smape_errors = []
    for num in smape_errors_file_object:
        if num == "NA\n" or num == "NA":
            current_model_all_smape_errors.append(np.nan)
        else:
            current_model_all_smape_errors.append(float(num))

    current_model_all_mase_errors = []
    for num in mase_errors_file_object:
        if num == "NA\n" or num == "NA":
            current_model_all_mase_errors.append(np.nan)
        else:
            current_model_all_mase_errors.append(float(num))

    smape_errors = np.asarray(current_model_all_smape_errors)
    mase_errors = np.asarray(current_model_all_mase_errors)

    # store the errors to calculate ranked errors later
    average_smape_errors_df[file_name_part] = smape_errors
    average_mase_errors_df[file_name_part] = mase_errors

    # calculate the mean, median
    mean_SMAPE = np.mean(smape_errors)
    median_SMAPE = np.median(smape_errors)

    mean_MASE = np.mean(mase_errors)
    median_MASE = np.median(mase_errors)

    all_errors_df.loc[-1] = [file_name_part, mean_SMAPE, median_SMAPE, mean_MASE, median_MASE]
    all_errors_df.index = all_errors_df.index + 1


# calculate the ranked errors
all_smape_ranks_df = np.mean(average_smape_errors_df.rank(axis=1), axis=0)
all_mase_ranks_df = np.mean(average_mase_errors_df.rank(axis=1), axis=0)

# add the ranked errors to all_errors_df
all_errors_df["Ranked_SMAPE"] = all_smape_ranks_df.tolist()
all_errors_df["Ranked_MASE"] = all_mase_ranks_df.tolist()

# write the errors to csv file
all_errors_df.to_csv(output_file, index=False)
