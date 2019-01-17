## This file concatenates the error results of the different clusters for a given dataset
## The script requires that all the files(mean_median file, all_smape_errors and all_mase_errors files) are present for all the clusters subject to consideration

import glob
import numpy as np
import argparse
import re
from collections import defaultdict

input_path = '../results/errors/'
output_path = '../results/errors/merged_cluster_results/'

# get the different cluster names as external arguments
argument_parser = argparse.ArgumentParser("Concatenate different cluster results of the same dataset")
argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
# argument_parser.add_argument('--model_seed_number', required=True, help='In case there are multiple results of the same model with different seeds')

# parse the user arguments
args = argument_parser.parse_args()

dataset_name = args.dataset_name
# seed = int(args.model_seed_number)

output_file_mean_median = output_path + "mean_median_" + dataset_name + "_"
output_file_all_smape_errors = output_path + "all_smape_errors_" + dataset_name + "_"
output_file_all_mase_errors = output_path + "all_mase_errors_" + dataset_name + "_"

# get the list of all the files matching the regex

# smape errors
all_smape_errors_files = [filename for filename in glob.iglob(input_path + "all_smape_errors_" + dataset_name + "_*")]

# mase errors
all_mase_errors_files = [filename for filename in glob.iglob(input_path + "all_mase_errors_" + dataset_name + "_*")]

all_mase_errors_dic = defaultdict(list)
all_smape_errors_dic = defaultdict(list)

# read the files one by one and merge the content
for smape_errors_file, mase_errors_file in zip(sorted(all_smape_errors_files),
                                                         sorted(all_mase_errors_files)):
    filename_smape_object = open(smape_errors_file)
    filename_mase_object = open(mase_errors_file)

    filename_part = re.split(pattern="all_smape_errors_" + dataset_name + "_" + "[a-zA-Z0-9]+_" , string=smape_errors_file, maxsplit=1)[1]

    # try:
    #     model_seed = int(filename_part.rsplit('_', 1)[1].split('.')[0])
    #     if model_seed != seed:
    #         continue
    # except:
    #     pass

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

    all_mase_errors_dic[filename_part] = all_mase_errors_dic[filename_part] + all_mase_errors
    all_smape_errors_dic[filename_part] = all_smape_errors_dic[filename_part] + all_smape_errors

# iterate the dictionaries and write to files
for (smape_key, smape_errors), (mase_key, mase_errors) in zip(all_smape_errors_dic.items(), all_mase_errors_dic.items()):

    # open the all errors smape file
    output_file_all_smape_errors_object = open(output_file_all_smape_errors + smape_key, "w")

    # open the all errors mase file
    output_file_all_mase_errors_object = open(output_file_all_mase_errors + mase_key, "w")

    # open the mean errors file
    # output_file_mean_median_object = open(output_file_mean_median + smape_key, "w")

    # calculate smape mean, std
    # smape_mean = np.mean(smape_errors, axis=0)
    # smape_median = np.median(smape_errors, axis=0)
    # smape_std = np.std(smape_errors, axis=0)

    # calculate mase mean, std
    # mase_mean = np.mean(mase_errors, axis=0)
    # mase_median = np.median(mase_errors, axis=0)
    # mase_std = np.std(mase_errors, axis=0)

    # write to files
    output_file_all_smape_errors_object.writelines('\n'.join(str(element) for element in smape_errors))
    output_file_all_mase_errors_object.writelines('\n'.join(str(element) for element in mase_errors))
    # output_file_mean_median_object.writelines("mean_SMAPE:" + str(smape_mean) + "\n")
    # output_file_mean_median_object.writelines("median_SMAPE:" + str(smape_median) + "\n")
    # output_file_mean_median_object.writelines("std_SMAPE:" + str(smape_std) + "\n")

    # output_file_mean_median_object.writelines("")
    # output_file_mean_median_object.writelines("")

    # output_file_mean_median_object.writelines("mean_MASE:" + str(mase_mean) + "\n")
    # output_file_mean_median_object.writelines("median_MASE:" + str(mase_median) + "\n")
    # output_file_mean_median_object.writelines("std_MASE:" + str(mase_std))

    # close the files
    output_file_all_smape_errors_object.close()
    output_file_all_mase_errors_object.close()
    # output_file_mean_median_object.close()
