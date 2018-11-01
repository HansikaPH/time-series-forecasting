## This file concatenates the results across all the models for the same dataset and writes the mean SMAPE, median SMAPE, ranked SMAPE, mean MASE, median MASE and ranked MASE to a file

import glob
import csv
import argparse

# get the different cluster names as external arguments
argument_parser = argparse.ArgumentParser("Create error summaries")
argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
argument_parser.add_argument('--is_merged_cluster_result', required=True, help='1/0 denoting whether the results are merged from multiple clusters or not')

# parse the user arguments
args = argument_parser.parse_args()
dataset_name = args.dataset_name
is_merged_cluster_result = args.is_merged_cluster_result


if int(is_merged_cluster_result):
    input_path = '../results/errors/merged_cluster_results'
else:
    input_path = '../results/errors/'

output_path = '../results/errors/aggregate_errors/'

output_file = output_path + dataset_name + ".csv"

output_file__object = open(output_file, "w")
csv_writer = csv.writer(output_file__object, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

# write the header row
csv_writer.writerow(["Model_Name", "Mean_SMAPE", "Median_SMAPE", "Ranked_SMAPE", "Mean_MASE", "Median_MASE", "Ranked_MASE"])

# get the list of all the files matching the regex except the destination file
mean_median_files = [filename for filename in glob.iglob(input_path + "mean_median_" + dataset_name + "_*")]
all_SMAPE_files = [filename for filename in glob.iglob(input_path + "all_smape_errors_" + dataset_name + "_*")]
all_MASE_files = [filename for filename in glob.iglob(input_path + "all_mase_errors_" + dataset_name + "_*")]


for smape_errors_file, mase_errors_file, mean_median_errors_file in zip(sorted(all_SMAPE_files),
                                                         sorted(all_MASE_files), sorted(mean_median_files)):
    smape_errors_file_object = open(smape_errors_file, "r")
    mase_errors_file_object = open(mase_errors_file, "r")
    mean_median_errors_file_object = open(mean_median_errors_file, "r")

    current_model_all_errors = [float(num) for num in smape_errors_file_object]

    # calculate the ranked smape

    # calculate the ranked mase

# list to store all the errors from all the models
# all_models_all_errors_df = pd.DataFrame()
#
# # read the files one by one and merge the content
# for filename in sorted(mean_median_files):
#     with open(filename) as file_object:
#         # read content from results file
#         content = file_object.readlines()
#
#         # write content to final output file
#         output_file_mean_median_object.writelines(filename + "\n")
#         output_file_mean_median_object.writelines(content)
#         output_file_mean_median_object.writelines("\n\n")
#         file_object.close()
#
# output_file_mean_median_object.close()
#
# # read the files one by one for all errors of models and concat into a list
# for filename in sorted(all_errors_files):
#     with open(filename) as file_object:
#
#         # read the numbers from all_errors file
#         current_model_all_errors = [float(num) for num in file_object]
#         all_models_all_errors_df[filename] = current_model_all_errors
#         file_object.close()
#
# output_file_mean_median_object.close()
#
# # calculate the ranked errors
# ranks_df = all_models_all_errors_df.rank(axis=1)
# ranked_errors_df = ranks_df.mean(axis=0)
# ranked_errors_df.to_csv(output_file_ranked_error)
