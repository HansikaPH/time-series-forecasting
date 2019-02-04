import re

def read_optimal_hyperparameter_values(file_name):
    # define dictionary to store the hyperparameter values
    hyperparameter_values_dic = {}

    with open(file_name) as configs_file:
        configs = configs_file.readlines()
        for config in configs:
            if not config.startswith('#') and config.strip():
                values = [value.strip() for value in (re.split(">>>", config))]
                hyperparameter_values_dic[values[0]] = float(values[1])

        configs_file.close()

    return hyperparameter_values_dic

def read_initial_hyperparameter_values(initial_hyperparameter_values_file):
    # define dictionary to store the hyperparameter values
    hyperparameter_values_dic = {}

    with open(initial_hyperparameter_values_file) as configs_file:
        configs = configs_file.readlines()
        for config in configs:
            if not config.startswith('#') and config.strip():
                values = [value.strip() for value in (re.split("-|,", config))]
                hyperparameter_values_dic[values[0]] = [float(values[1]), float(values[2])]

        configs_file.close()

    return hyperparameter_values_dic