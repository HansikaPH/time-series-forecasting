## This file concatenates the optimal hyperparameter configs across all the models for the same dataset and writes to a csv file

import glob
import argparse
import pandas as pd
import re
from utility_scripts.hyperparameter_scripts.hyperparameter_config_reader import read_optimal_hyperparameter_values

# get the different cluster names as external arguments
argument_parser = argparse.ArgumentParser("Create hyperparameter summaries")
argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')

# parse the user arguments
args = argument_parser.parse_args()
dataset_name = args.dataset_name

input_path = '../results/optimized_configurations/'

output_path = '../results/optimized_configurations/aggregate_hyperparameter_configs/'

output_file = output_path + dataset_name + ".csv"

# get the list of all the files matching the regex
hyperparameter_files = [filename for filename in glob.iglob(input_path + dataset_name + "_*")]

hyperparameters_df = pd.DataFrame(
    columns=["Model_Name", "cell_dimension", "gaussian_noise_stdev", "l2_regularization", "max_epoch_size",
             "max_num_epochs", "minibatch_size", "num_hidden_layers", "random_normal_initializer_stdev",
             "rate_of_learning"])

# concat all the hyperparameters to data frames
for config_file in sorted(hyperparameter_files):
    file_name_part = re.split(pattern=dataset_name + "_", string=config_file, maxsplit=1)[1]

    model_name = file_name_part.rsplit('_', 1)[0]

    print(model_name)

    hyperparameter_values_dic = read_optimal_hyperparameter_values(config_file)
    if "rate_of_learning" not in hyperparameter_values_dic.keys():
        hyperparameter_values_dic["rate_of_learning"] = "-"

    hyperparameters_df.loc[-1] = [model_name, hyperparameter_values_dic["cell_dimension"],
                                  hyperparameter_values_dic["gaussian_noise_stdev"],
                                  hyperparameter_values_dic["l2_regularization"],
                                  hyperparameter_values_dic["max_epoch_size"],
                                  hyperparameter_values_dic["max_num_epochs"],
                                  hyperparameter_values_dic["minibatch_size"],
                                  hyperparameter_values_dic["num_hidden_layers"],
                                  hyperparameter_values_dic["random_normal_initializer_stdev"],
                                  hyperparameter_values_dic["rate_of_learning"]]
    hyperparameters_df.index = hyperparameters_df.index + 1

# write the errors to csv file
hyperparameters_df.to_csv(output_file, index=False)
