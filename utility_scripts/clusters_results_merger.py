## This file concatenates the error results of the different clusters for a given dataset

import glob
import numpy as np
import argparse

input_path = '../results/errors/'
output_path = '../results/errors/merged_cluster_results'

# get the different cluster names as external arguments
argument_parser = argparse.ArgumentParser("Concatenate different cluster results of the same dataset")
argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')

# parse the user arguments
args = argument_parser.parse_args()

dataset_name = args.dataset_name

output_file_mean_median = input_path + "mean_median_" + dataset_name + "_"
output_file_all_smape_errors = input_path + "all_smape_errors_" + dataset_name + "_"
output_file_all_mase_errors = input_path + "all_mase_errors_" + dataset_name + "_"

# get the list of all the files matching the regex

# smape errors
all_smape_errors_files = [filename for filename in glob.iglob(input_path + "all_smape_errors_" + dataset_name + "_*")]

# mase errors
all_mase_errors_files = [filename for filename in glob.iglob(input_path + "all_mase_errors_" + dataset_name + "_*")]

all_mase_errors_dic = {}
all_smape_errors_dic = {}

# read the files one by one and merge the content
for smape_errors_file, mase_errors_file in zip(sorted(all_smape_errors_files),
                                                         sorted(all_mase_errors_files)):
    filename_smape_object = open(smape_errors_file)
    filename_mase_object = open(mase_errors_file)

    # TODO: check the regular expression
    filename_part = smape_errors_file.split(sep="all_smape_errors_" + dataset_name + "[a-zA-Z0-9]+$_" , maxsplit=1)[1]

    # read the errors from both files
    all_smape_errors = [float(num) for num in filename_smape_object]
    all_mase_errors = [float(num) for num in filename_mase_object]

    all_mase_errors_dic[filename_part].append(all_mase_errors)
    all_smape_errors_dic[filename_part].appenf(all_smape_errors)

# iterate the dictionaries and write to files
for (smape_key, smape_errors), (mase_key, mase_errors) in zip(all_mase_errors_dic.items(), all_smape_errors_dic.items()):

    # open the all errors smape file
    output_file_all_smape_errors_object = open(output_file_all_smape_errors + dataset_name + "_" + smape_key, "w")

    # open the all errors mase file
    output_file_all_mase_errors_object = open(output_file_all_mase_errors + dataset_name + "_" + mase_key, "w")

    # open the mean errors file
    output_file_mean_median_object = open(output_file_mean_median + dataset_name + "_" + smape_key, "w")

    # calculate smape mean, std
    smape_mean = np.mean(smape_errors, axis=0)
    smape_median = np.median(smape_errors, axis=0)
    smape_std = np.std(smape_errors, axis=0)

    # calculate mase mean, std
    mase_mean = np.mean(mase_errors, axis=0)
    mase_median = np.median(mase_errors, axis=0)
    mase_std = np.std(mase_errors, axis=0)

    # write to files
    output_file_all_smape_errors_object.writelines(smape_errors)
    output_file_all_mase_errors_object.writelines(mase_errors)
    output_file_mean_median_object.writelines("mean_SMAPE:" + smape_mean)
    output_file_mean_median_object.writelines("median_SMAPE:" + smape_median)
    output_file_mean_median_object.writelines("std_SMAPE:" + smape_std)

    output_file_mean_median_object.writelines("")
    output_file_mean_median_object.writelines("")

    output_file_mean_median_object.writelines("mean_MASE:" + mase_mean)
    output_file_mean_median_object.writelines("median_MASE:" + mase_median)
    output_file_mean_median_object.writelines("std_MASE:" + mase_std)

    # close the files
    output_file_all_smape_errors_object.close()
    output_file_all_mase_errors_object.close()
    output_file_mean_median_object.close()
