## This file concatenates the error results of the different clusters for a given dataset
## The script requires that all the files(mean_median file, all_smape_errors and all_mase_errors files) are present for all the clusters subject to consideration

import glob
import numpy as np
import argparse
import re
from collections import defaultdict

input_path = '../results/ensemble_errors/'
output_path = '../results/ensemble_errors/merged_cluster_results/'

# get the different cluster names as external arguments
argument_parser = argparse.ArgumentParser("Concatenate different cluster results of the same dataset")
argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')

# parse the user arguments
args = argument_parser.parse_args()

dataset_name = args.dataset_name

output_file_mean_median = output_path + "mean_median_" + dataset_name + "_"
output_file_all_smape_errors = output_path + "all_smape_errors_" + dataset_name + "_"
output_file_all_mase_errors = output_path + "all_mase_errors_" + dataset_name + "_"

# get the list of all the files matching the regex

# smape errors
all_smape_errors_files = [filename for filename in glob.iglob(input_path + "all_smape_errors_" + dataset_name + "_*")]

# mase errors
all_mase_errors_files = [filename for filename in glob.iglob(input_path + "all_mase_errors_" + dataset_name + "_*")]

all_mase_errors_dic = defaultdict(dict)
all_smape_errors_dic = defaultdict(dict)

category_order_dic = {
    "O12": 0,
    "O6": 1
}
# read the files one by one and merge the content
for smape_errors_file, mase_errors_file in zip(sorted(all_smape_errors_files),
                                                         sorted(all_mase_errors_files)):
    filename_smape_object = open(smape_errors_file)
    filename_mase_object = open(mase_errors_file)

    string_list = re.split(pattern="all_smape_errors_" + dataset_name + "_" + "[a-zA-Z0-9]+_" , string=smape_errors_file, maxsplit=1)
    filename_part = string_list[1]
    category = re.search(dataset_name + '_([a-zA-Z0-9]+)_', smape_errors_file).group(1)

    # read the errors from both files
    all_smape_errors = []
    for num in filename_smape_object:
        if num == "NA\n":
            all_smape_errors.append("NA")
        else:
            all_smape_errors.append(float(num))

    all_mase_errors = []
    for num in filename_mase_object:
        if num == "NA\n":
            all_mase_errors.append("NA")
        else:
            all_mase_errors.append(float(num))

    all_mase_errors_dic[filename_part][category] = all_mase_errors
    all_smape_errors_dic[filename_part][category] = all_smape_errors

# iterate the dictionaries and write to files
for (smape_key, smape_errors_dic), (mase_key, mase_errors_dic) in zip(all_smape_errors_dic.items(), all_mase_errors_dic.items()):

    # open the all errors smape file
    output_file_all_smape_errors_object = open(output_file_all_smape_errors + smape_key, "a")

    # open the all errors mase file
    output_file_all_mase_errors_object = open(output_file_all_mase_errors + mase_key, "a")


    # sort the categories according to the given order
    for (category, index) in sorted(category_order_dic.items(), key=lambda item: item[1]):
        category_smape_errors = smape_errors_dic[category]
        category_mase_errors = mase_errors_dic[category]

        # write to files
        output_file_all_smape_errors_object.writelines('\n'.join(str(element) for element in category_smape_errors))
        output_file_all_smape_errors_object.writelines('\n')
        output_file_all_mase_errors_object.writelines('\n'.join(str(element) for element in category_mase_errors))
        output_file_all_mase_errors_object.writelines('\n')


    # close the files
    output_file_all_smape_errors_object.close()
    output_file_all_mase_errors_object.close()
