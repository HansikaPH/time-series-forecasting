## This file concatenates the results across all the models for the same dataset and writes the mean SMAPE, median SMAPE, ranked SMAPE, mean MASE, median MASE and ranked MASE to a file

# TODO: different executions with random seeds

import glob
import sys
import pandas as pd

dataset_name = sys.argv[1]

output_path = '../results/errors/aggregate_errors/'
input_path = '../results/errors/'

# output_file_mean_median = output_path + "mean_median_" + dataset_name
# output_file_ranked_error = output_path + "ranked_error_" + dataset_name
output_file = output_path + dataset_name + ".csv"

output_file__object = open(output_file, "w")
# output_file_mean_median_object = open(output_file_mean_median, "w")
# output_file_ranked_error_object = open(output_file_ranked_error, "w")

# get the list of all the files matching the regex except the destination file
mean_median_files = [filename for filename in glob.iglob(input_path + "mean_median_" + dataset_name + "_*") if "O12" not in filename or "O6" not in filename]
all_SMAPE_files = [filename for filename in glob.iglob(input_path + "all_smape_errors_" + dataset_name + "_*") if "O12" not in filename or "O6" not in filename]
all_MASE_files = [filename for filename in glob.iglob(input_path + "all_mase_errors_" + dataset_name + "_*") if "O12" not in filename or "O6" not in filename]

# list to store all the errors from all the models
all_models_all_errors_df = pd.DataFrame()

# read the files one by one and merge the content
for filename in sorted(mean_median_files):
    with open(filename) as file_object:
        # read content from results file
        content = file_object.readlines()

        # write content to final output file
        output_file_mean_median_object.writelines(filename + "\n")
        output_file_mean_median_object.writelines(content)
        output_file_mean_median_object.writelines("\n\n")
        file_object.close()

output_file_mean_median_object.close()

# read the files one by one for all errors of models and concat into a list
for filename in sorted(all_errors_files):
    with open(filename) as file_object:

        # read the numbers from all_errors file
        current_model_all_errors = [float(num) for num in file_object]
        all_models_all_errors_df[filename] = current_model_all_errors
        file_object.close()

output_file_mean_median_object.close()

# calculate the ranked errors
ranks_df = all_models_all_errors_df.rank(axis=1)
ranked_errors_df = ranks_df.mean(axis=0)
ranked_errors_df.to_csv(output_file_ranked_error)
