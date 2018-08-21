import numpy as np
import tensorflow as tf
import argparse
from bayes_opt import BayesianOptimization
from persist_optimized_config_results import persist_results
from generic_model_tester import testing
import re

# import the config space and the different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

#import SMAC utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

# import the different model types
from stacking_model.stacking_model_trainer import StackingModelTrainer
from seq2seq_model.non_moving_window.seq2seq_model_trainer import Seq2SeqModelTrainer
from attention_model.attention_model_trainer import AttentionModelTrainer

# import the cocob optimizer
from external_packages import cocob_optimizer

LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

optimized_config_directory = 'results/optimized_configurations/'
learning_rate = 0.0

# function to create the optimizer
def adagrad_optimizer_fn(total_loss):
    return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)

def adam_optimizer_fn(total_loss):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

def cocob_optimizer_fn(total_loss):
    return cocob_optimizer.COCOB().minimize(loss=total_loss)

def read_initial_hyperparameter_configurations():
    # define dictionary to store the hyperparameter values
    hyperparameter_configs_dic = {}

    with open(initial_hyperparameter_configs_file) as configs_file:
        configs = configs_file.readlines()
        for config in configs:
            if config.strip():
                values = [value.strip() for value in (re.split("-|,", config))]
                hyperparameter_configs_dic[values[0]] = [float(values[1]), float(values[2])]

        configs_file.close()

    return hyperparameter_configs_dic

# Training the time series
def train_model_smac(configs):

    rate_of_learning = configs["rate_of_learning"]
    lstm_cell_dimension = configs["lstm_cell_dimension"]
    num_hidden_layers = configs["num_hidden_layers"]
    minibatch_size = configs["minibatch_size"]
    max_epoch_size = configs["max_epoch_size"]
    max_num_epochs = configs["max_num_epochs"]
    l2_regularization = configs["l2_regularization"]
    gaussian_noise_stdev = configs["gaussian_noise_stdev"]

    global learning_rate
    learning_rate = rate_of_learning

    print(configs)

    # select the appropriate type of optimizer
    error =  model_trainer.train_model(num_hidden_layers = num_hidden_layers,
                        lstm_cell_dimension = lstm_cell_dimension,
                        minibatch_size = minibatch_size,
                        max_epoch_size = max_epoch_size,
                        max_num_epochs = max_num_epochs,
                        l2_regularization = l2_regularization,
                        gaussian_noise_stdev = gaussian_noise_stdev,
                        optimizer_fn = optimizer_fn)

    print("Error: {}".format(error))
    return error

def train_model_bayesian(num_hidden_layers, lstm_cell_dimension, minibatch_size, max_epoch_size, max_num_epochs, l2_regularization, gaussian_noise_stdev, rate_of_learning = 0.0):
    global learning_rate
    learning_rate = rate_of_learning

    error = model_trainer.train_model(num_hidden_layers=int(round(num_hidden_layers)),
                                      lstm_cell_dimension=int(round(lstm_cell_dimension)),
                                      minibatch_size=int(round(minibatch_size)),
                                      max_epoch_size=int(round(max_epoch_size)),
                                      max_num_epochs=int(round(max_num_epochs)),
                                      l2_regularization=l2_regularization,
                                      gaussian_noise_stdev=gaussian_noise_stdev,
                                      optimizer_fn=optimizer_fn)
    print("Error: {}".format(error))
    return -1 * error


def bayesian_optimization():
    init_points = 2
    num_iter = 30

    parameters = {'num_hidden_layers': (hyperparameter_configs_dic['num_hidden_layers'][0], hyperparameter_configs_dic['num_hidden_layers'][1]),
                  'lstm_cell_dimension': (hyperparameter_configs_dic['lstm_cell_dimension'][0], hyperparameter_configs_dic['lstm_cell_dimension'][1]),
                  'minibatch_size': (hyperparameter_configs_dic['minibatch_size'][0], hyperparameter_configs_dic['minibatch_size'][1]),
                  'max_epoch_size': (hyperparameter_configs_dic['max_epoch_size'][0], hyperparameter_configs_dic['max_epoch_size'][1]),
                  'max_num_epochs': (hyperparameter_configs_dic['max_num_epochs'][0], hyperparameter_configs_dic['max_num_epochs'][1]),
                  'l2_regularization': (hyperparameter_configs_dic['l2_regularization'][0], hyperparameter_configs_dic['l2_regularization'][1]),
                  'gaussian_noise_stdev': (hyperparameter_configs_dic['gaussian_noise_stdev'][0], hyperparameter_configs_dic['gaussian_noise_stdev'][1])
                  }

    # adding the hyperparameter for learning rate if the optimization is not cocob
    if optimizer != 'cocob':
        parameters['rate_of_learning'] = (hyperparameter_configs_dic["rate_of_learning"][0], hyperparameter_configs_dic["rate_of_learning"][1])

    # using bayesian optimizer for hyperparameter optimization
    bayesian_optimization = BayesianOptimization(train_model_bayesian, parameters)

    bayesian_optimization.maximize(init_points=init_points, n_iter=num_iter)
    optimized_configuration = bayesian_optimization.res['max']['max_params']
    print(optimized_configuration)

    return optimized_configuration

def smac():

    # Build Configuration Space which defines all parameters and their ranges
    configuration_space = ConfigurationSpace()

    rate_of_learning = UniformFloatHyperparameter("rate_of_learning", hyperparameter_configs_dic['rate_of_learning'][0], hyperparameter_configs_dic['rate_of_learning'][1],
                                                  default_value = hyperparameter_configs_dic['rate_of_learning'][0])
    lstm_cell_dimension = UniformIntegerHyperparameter("lstm_cell_dimension", hyperparameter_configs_dic['lstm_cell_dimension'][0], hyperparameter_configs_dic['lstm_cell_dimension'][1],
                                                       default_value = hyperparameter_configs_dic['lstm_cell_dimension'][0])
    no_hidden_layers = UniformIntegerHyperparameter("num_hidden_layers", hyperparameter_configs_dic['num_hidden_layers'][0], hyperparameter_configs_dic['num_hidden_layers'][1],
                                                    default_value = hyperparameter_configs_dic['num_hidden_layers'][0])
    minibatch_size = UniformIntegerHyperparameter("minibatch_size", hyperparameter_configs_dic['minibatch_size'][0], hyperparameter_configs_dic['minibatch_size'][1],
                                                  default_value = hyperparameter_configs_dic['minibatch_size'][0])
    max_epoch_size = UniformIntegerHyperparameter("max_epoch_size", hyperparameter_configs_dic['max_epoch_size'][0], hyperparameter_configs_dic['max_epoch_size'][1],
                                                  default_value = hyperparameter_configs_dic['max_epoch_size'][0])
    max_num_of_epochs = UniformIntegerHyperparameter("max_num_epochs", hyperparameter_configs_dic['max_num_epochs'][0], hyperparameter_configs_dic['max_num_epochs'][1],
                                                     default_value = hyperparameter_configs_dic['max_num_epochs'][0])
    l2_regularization = UniformFloatHyperparameter("l2_regularization", hyperparameter_configs_dic['l2_regularization'][0], hyperparameter_configs_dic['l2_regularization'][1],
                                                   default_value = hyperparameter_configs_dic['l2_regularization'][0])
    gaussian_noise_stdev = UniformFloatHyperparameter("gaussian_noise_stdev", hyperparameter_configs_dic['gaussian_noise_stdev'][0], hyperparameter_configs_dic['gaussian_noise_stdev'][1],
                                                      default_value = hyperparameter_configs_dic['gaussian_noise_stdev'][0])


    # add the hyperparameter for learning rate only if the  optimization is not cocob
    if optimizer == "cocob":
        configuration_space.add_hyperparameters([lstm_cell_dimension, no_hidden_layers, minibatch_size, max_epoch_size, max_num_of_epochs,
                                             l2_regularization, gaussian_noise_stdev])
    else:
        configuration_space.add_hyperparameters([rate_of_learning, lstm_cell_dimension, no_hidden_layers, minibatch_size, max_epoch_size, max_num_of_epochs,
             l2_regularization, gaussian_noise_stdev])

    # creating the scenario object
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": 50,
        "cs": configuration_space,
        "deterministic": True,
        "output_dir": "Logs"
    })

    # optimize using an SMAC object
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(0), tae_runner=train_model_smac)

    incumbent = smac.optimize()
    smape_error = train_model_smac(incumbent)

    print("Optimized configuration: {}".format(incumbent))
    print("Optimized Value: {}\n".format(smape_error))
    return incumbent.get_dictionary()


if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser("Train different forecasting models")
    argument_parser.add_argument('--dataset_name', required = True, help = 'Unique string for the name of the dataset')
    argument_parser.add_argument('--contain_zero_values', required = True, help = 'Whether the dataset contains zero values')
    argument_parser.add_argument('--initial_hyperparameter_configs_file', required=True, help='The file for the initial hyperparameter configurations')
    argument_parser.add_argument('--binary_train_file', required = True, help = 'The tfrecords file for train dataset')
    argument_parser.add_argument('--binary_valid_file', required=True, help='The tfrecords file for validation dataset')
    argument_parser.add_argument('--binary_test_file', required=True, help='The tfrecords file for test dataset')
    argument_parser.add_argument('--txt_test_file', required=True, help='The txt file for test dataset')
    argument_parser.add_argument('--actual_results_file', required=True, help='The txt file of the actual results')
    argument_parser.add_argument('--input_size', required=False, help='The input size of the moving window')
    argument_parser.add_argument('--subsequence_length', required=False, help='The subsequence length to use truncated backpropagation')
    argument_parser.add_argument('--forecast_horizon', required=True, help='The forecast horizon of the dataset')
    argument_parser.add_argument('--optimizer', required = True, help = 'The type of the optimizer(cocob/adam/adagrad...)')
    argument_parser.add_argument('--hyperparameter_tuning', required=True, help='The method for hyperparameter tuning(bayesian/smac)')
    argument_parser.add_argument('--model_type', required=True, help='The type of the model(stacking/non_moving_window/attention)')

    # parse the user arguments
    args = argument_parser.parse_args()

    dataset_name = args.dataset_name
    initial_hyperparameter_configs_file = args.initial_hyperparameter_configs_file
    binary_train_file_path = args.binary_train_file
    binary_validation_file_path = args.binary_valid_file
    contain_zero_values = args.contain_zero_values
    if(args.input_size):
        input_size = int(args.input_size)
    else:
        input_size = 0
    if(args.subsequence_length):
        subsequence_length = int(args.subsequence_length)
    else:
        subsequence_length = 0
    output_size = int(args.forecast_horizon)
    optimizer = args.optimizer
    hyperparameter_tuning = args.hyperparameter_tuning
    model_type = args.model_type

    print("Model Training Started for {}_{}_{}_{}".format(dataset_name, model_type, hyperparameter_tuning, optimizer))

    # select the optimizer
    if optimizer == "cocob":
        optimizer_fn = cocob_optimizer_fn
    elif optimizer == "adagrad":
        optimizer_fn = adagrad_optimizer_fn
    elif optimizer == "adam":
        optimizer_fn = adam_optimizer_fn

    # select the model type
    if model_type == "stacking":
        model_trainer = StackingModelTrainer(
            use_bias = BIAS,
            use_peepholes = LSTM_USE_PEEPHOLES,
            input_size = input_size,
            output_size = output_size,
            binary_train_file_path = binary_train_file_path,
            binary_validation_file_path = binary_validation_file_path,
            contain_zero_values = contain_zero_values
        )
    elif model_type == "seq2seq":
        model_trainer = Seq2SeqModelTrainer(
            use_bias=BIAS,
            use_peepholes=LSTM_USE_PEEPHOLES,
            subsequence_length = subsequence_length,
            output_size=output_size,
            binary_train_file_path=binary_train_file_path,
            binary_validation_file_path=binary_validation_file_path,
            contain_zero_values=contain_zero_values
        )
    elif model_type == "attention":
        model_trainer = AttentionModelTrainer(
            use_bias=BIAS,
            use_peepholes=LSTM_USE_PEEPHOLES,
            output_size=output_size,
            binary_train_file_path=binary_train_file_path,
            binary_validation_file_path=binary_validation_file_path,
            contain_zero_values=contain_zero_values
        )

    # read the initial hyperparamter configurations from the file
    hyperparameter_configs_dic = read_initial_hyperparameter_configurations()

    # select the hyperparameter tuning method
    if hyperparameter_tuning == "bayesian":
        optimized_configuration = bayesian_optimization()
    elif hyperparameter_tuning == "smac":
        optimized_configuration = smac()

    # persist the optimized configuration to a file
    persist_results(optimized_configuration, optimized_config_directory + '/' + dataset_name + '_' + model_type + '_' + hyperparameter_tuning + '_' + optimizer + '.txt')

    # test the model
    testing(args, optimized_configuration)
