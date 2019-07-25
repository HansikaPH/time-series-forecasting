import numpy as np
import tensorflow as tf
import argparse
from utility_scripts.persist_optimized_config_results import persist_results
from generic_model_tester import testing
from utility_scripts.hyperparameter_scripts.hyperparameter_config_reader import read_initial_hyperparameter_values

# import the config space and the different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

# import SMAC utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

## import the different model architectures

# stacking model
from rnn_architectures.stacking_model.stacking_model_trainer import \
    StackingModelTrainer as StackingModelTrainer

# seq2seq model with decoder
from rnn_architectures.seq2seq_model.with_decoder.non_moving_window.unaccumulated_error.seq2seq_model_trainer import \
    Seq2SeqModelTrainer as Seq2SeqModelTrainerWithNonMovingWindowUnaccumulatedError
from rnn_architectures.seq2seq_model.with_decoder.non_moving_window.accumulated_error.seq2seq_model_trainer import \
    Seq2SeqModelTrainer as Seq2SeqModelTrainerWithNonMovingWindowAccumulatedError

# seq2seq model with dense layer
from rnn_architectures.seq2seq_model.with_dense_layer.non_moving_window.unaccumulated_error.seq2seq_model_trainer import \
    Seq2SeqModelTrainerWithDenseLayer as Seq2SeqModelTrainerWithDenseLayerNonMovingWindowUnaccumulatedError
from rnn_architectures.seq2seq_model.with_dense_layer.non_moving_window.accumulated_error.seq2seq_model_trainer import \
    Seq2SeqModelTrainerWithDenseLayer as Seq2SeqModelTrainerWithDenseLayerNonMovingWindowAccumulatedError
from rnn_architectures.seq2seq_model.with_dense_layer.moving_window.unaccumulated_error.seq2seq_model_trainer import \
    Seq2SeqModelTrainerWithDenseLayer as Seq2SeqModelTrainerWithDenseLayerMovingWindow

# import the cocob optimizer
from external_packages import cocob_optimizer

from configs.global_configs import hyperparameter_tuning_configs
from configs.global_configs import model_training_configs

import csv

LSTM_USE_PEEPHOLES = True
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


# Training the time series
def train_model_smac(configs):
    error, _ = train_model(configs)
    return error

# final execution with the optimized config
def train_model(configs):
    if "rate_of_learning" in configs.keys():
        rate_of_learning = configs["rate_of_learning"]
        global learning_rate
        learning_rate = rate_of_learning
    cell_dimension = configs["cell_dimension"]
    num_hidden_layers = configs["num_hidden_layers"]
    minibatch_size = configs["minibatch_size"]
    max_epoch_size = configs["max_epoch_size"]
    max_num_epochs = configs["max_num_epochs"]
    l2_regularization = configs["l2_regularization"]
    gaussian_noise_stdev = configs["gaussian_noise_stdev"]
    random_normal_initializer_stdev = configs["random_normal_initializer_stdev"]

    print(configs)

    # select the appropriate type of optimizer
    error, error_list = model_trainer.train_model(num_hidden_layers=num_hidden_layers,
                                      cell_dimension=cell_dimension,
                                      minibatch_size=minibatch_size,
                                      max_epoch_size=max_epoch_size,
                                      max_num_epochs=max_num_epochs,
                                      l2_regularization=l2_regularization,
                                      gaussian_noise_stdev=gaussian_noise_stdev,
                                      random_normal_initializer_stdev=random_normal_initializer_stdev,
                                      optimizer_fn=optimizer_fn)

    print(model_identifier)
    return error, error_list

def smac():
    # Build Configuration Space which defines all parameters and their ranges
    configuration_space = ConfigurationSpace()

    rate_of_learning = UniformFloatHyperparameter("rate_of_learning", hyperparameter_values_dic['rate_of_learning'][0],
                                                  hyperparameter_values_dic['rate_of_learning'][1],
                                                  default_value=hyperparameter_values_dic['rate_of_learning'][0])
    cell_dimension = UniformIntegerHyperparameter("cell_dimension",
                                                  hyperparameter_values_dic['cell_dimension'][0],
                                                  hyperparameter_values_dic['cell_dimension'][1],
                                                  default_value=hyperparameter_values_dic['cell_dimension'][
                                                      0])
    no_hidden_layers = UniformIntegerHyperparameter("num_hidden_layers",
                                                    hyperparameter_values_dic['num_hidden_layers'][0],
                                                    hyperparameter_values_dic['num_hidden_layers'][1],
                                                    default_value=hyperparameter_values_dic['num_hidden_layers'][0])
    minibatch_size = UniformIntegerHyperparameter("minibatch_size", hyperparameter_values_dic['minibatch_size'][0],
                                                  hyperparameter_values_dic['minibatch_size'][1],
                                                  default_value=hyperparameter_values_dic['minibatch_size'][0])
    max_epoch_size = UniformIntegerHyperparameter("max_epoch_size", hyperparameter_values_dic['max_epoch_size'][0],
                                                  hyperparameter_values_dic['max_epoch_size'][1],
                                                  default_value=hyperparameter_values_dic['max_epoch_size'][0])
    max_num_of_epochs = UniformIntegerHyperparameter("max_num_epochs", hyperparameter_values_dic['max_num_epochs'][0],
                                                     hyperparameter_values_dic['max_num_epochs'][1],
                                                     default_value=hyperparameter_values_dic['max_num_epochs'][0])
    l2_regularization = UniformFloatHyperparameter("l2_regularization",
                                                   hyperparameter_values_dic['l2_regularization'][0],
                                                   hyperparameter_values_dic['l2_regularization'][1],
                                                   default_value=hyperparameter_values_dic['l2_regularization'][0])
    gaussian_noise_stdev = UniformFloatHyperparameter("gaussian_noise_stdev",
                                                      hyperparameter_values_dic['gaussian_noise_stdev'][0],
                                                      hyperparameter_values_dic['gaussian_noise_stdev'][1],
                                                      default_value=hyperparameter_values_dic['gaussian_noise_stdev'][
                                                          0])
    random_normal_initializer_stdev = UniformFloatHyperparameter("random_normal_initializer_stdev",
                                                                 hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][0],
                                                                 hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][1],
                                                                 default_value=hyperparameter_values_dic[
                                                                     'random_normal_initializer_stdev'][
                                                                     0])

    # add the hyperparameter for learning rate only if the  optimization is not cocob
    if optimizer == "cocob":
        configuration_space.add_hyperparameters(
            [cell_dimension, no_hidden_layers, minibatch_size, max_epoch_size, max_num_of_epochs,
             l2_regularization, gaussian_noise_stdev, random_normal_initializer_stdev])
    else:

        configuration_space.add_hyperparameters(
            [rate_of_learning, cell_dimension, minibatch_size, max_epoch_size,
             max_num_of_epochs, no_hidden_layers,
             l2_regularization, gaussian_noise_stdev, random_normal_initializer_stdev])

    # creating the scenario object
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": hyperparameter_tuning_configs.SMAC_RUNCOUNT_LIMIT,
        "cs": configuration_space,
        "deterministic": "true",
        "abort_on_first_run_crash": "false"
    })

    # optimize using an SMAC object
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=train_model_smac)

    incumbent = smac.optimize()
    return incumbent.get_dictionary()


if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser("Train different forecasting models")
    argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
    argument_parser.add_argument('--contain_zero_values', required=True,
                                 help='Whether the dataset contains zero values(0/1)')
    argument_parser.add_argument('--address_near_zero_instability', required=False,
                                 help='Whether to use a custom SMAPE function to address near zero instability(0/1). Default is 0')
    argument_parser.add_argument('--integer_conversion', required=False,
                                 help='Whether to convert the final forecasts to integers(0/1). Default is 0')
    argument_parser.add_argument('--initial_hyperparameter_values_file', required=True,
                                 help='The file for the initial hyperparameter configurations')
    argument_parser.add_argument('--binary_train_file_train_mode', required=True,
                                 help='The tfrecords file for train dataset in the training mode')
    argument_parser.add_argument('--binary_valid_file_train_mode', required=True,
                                 help='The tfrecords file for validation dataset in the training mode')
    argument_parser.add_argument('--binary_train_file_test_mode', required=True,
                                 help='The tfrecords file for train dataset in the testing mode')
    argument_parser.add_argument('--binary_test_file_test_mode', required=True,
                                 help='The tfrecords file for test dataset in the testing mode')
    argument_parser.add_argument('--txt_test_file', required=True, help='The txt file for test dataset')
    argument_parser.add_argument('--actual_results_file', required=True, help='The txt file of the actual results')
    argument_parser.add_argument('--original_data_file', required=True, help='The txt file of the original dataset')
    argument_parser.add_argument('--cell_type', required=False,
                                 help='The cell type of the RNN(LSTM/GRU/RNN). Default is LSTM')
    argument_parser.add_argument('--input_size', required=False,
                                 help='The input size of the moving window. Default is 0')
    argument_parser.add_argument('--seasonality_period', required=True, help='The seasonality period of the time series')
    argument_parser.add_argument('--forecast_horizon', required=True, help='The forecast horizon of the dataset')
    argument_parser.add_argument('--optimizer', required=True, help='The type of the optimizer(cocob/adam/adagrad...)')
    argument_parser.add_argument('--hyperparameter_tuning', required=True,
                                 help='The method for hyperparameter tuning(bayesian/smac)')
    argument_parser.add_argument('--model_type', required=True,
                                 help='The type of the model(stacking/seq2seq/seq2seqwithdenselayer)')
    argument_parser.add_argument('--input_format', required=True, help='Input format(moving_window/non_moving_window)')
    argument_parser.add_argument('--without_stl_decomposition', required=False,
                                 help='Whether not to use stl decomposition(0/1). Default is 0')
    argument_parser.add_argument('--with_truncated_backpropagation', required=False,
                                 help='Whether not to use truncated backpropagation(0/1). Default is 0')
    argument_parser.add_argument('--with_accumulated_error', required=False,
                                 help='Whether to accumulate errors over the moving windows. Default is 0')
    argument_parser.add_argument('--seed', required=True, help='Integer seed to use as the random seed')

    # parse the user arguments
    args = argument_parser.parse_args()

    dataset_name = args.dataset_name
    initial_hyperparameter_values_file = args.initial_hyperparameter_values_file
    binary_train_file_path_train_mode = args.binary_train_file_train_mode
    binary_validation_file_path_train_mode = args.binary_valid_file_train_mode
    contain_zero_values = int(args.contain_zero_values)

    if args.input_size:
        input_size = int(args.input_size)
    else:
        input_size = 0

    output_size = int(args.forecast_horizon)
    optimizer = args.optimizer
    hyperparameter_tuning = args.hyperparameter_tuning
    model_type = args.model_type
    input_format = args.input_format
    seed = int(args.seed)

    if args.without_stl_decomposition:
        without_stl_decomposition = bool(int(args.without_stl_decomposition))
    else:
        without_stl_decomposition = False

    if args.with_truncated_backpropagation:
        with_truncated_backpropagation = bool(int(args.with_truncated_backpropagation))
    else:
        with_truncated_backpropagation = False

    if args.cell_type:
        cell_type = args.cell_type
    else:
        cell_type = "LSTM"

    if args.with_accumulated_error:
        with_accumulated_error = bool(int(args.with_accumulated_error))
    else:
        with_accumulated_error = False

    if args.address_near_zero_instability:
        address_near_zero_instability = bool(int(args.address_near_zero_instability))
    else:
        address_near_zero_instability = False

    if args.integer_conversion:
        integer_conversion = bool(int(args.integer_conversion))
    else:
        integer_conversion = False

    if with_truncated_backpropagation:
        tbptt_identifier = "with_truncated_backpropagation"
    else:
        tbptt_identifier = "without_truncated_backpropagation"

    if without_stl_decomposition:
        stl_decomposition_identifier = "without_stl_decomposition"
    else:
        stl_decomposition_identifier = "with_stl_decomposition"

    if with_accumulated_error:
        accumulated_error_identifier = "with_accumulated_error"
    else:
        accumulated_error_identifier = "without_accumulated_error"

    model_identifier = dataset_name + "_" + model_type + "_" + cell_type + "cell" + "_" + input_format + "_" + stl_decomposition_identifier + "_" + hyperparameter_tuning + "_" + optimizer + "_" + tbptt_identifier + "_" + accumulated_error_identifier + "_" + str(
        seed)
    print("Model Training Started for {}".format(model_identifier))

    # select the optimizer
    if optimizer == "cocob":
        optimizer_fn = cocob_optimizer_fn
    elif optimizer == "adagrad":
        optimizer_fn = adagrad_optimizer_fn
    elif optimizer == "adam":
        optimizer_fn = adam_optimizer_fn

    # define the key word arguments for the different model types
    model_kwargs = {
        'use_bias': BIAS,
        'use_peepholes': LSTM_USE_PEEPHOLES,
        'input_size': input_size,
        'output_size': output_size,
        'binary_train_file_path': binary_train_file_path_train_mode,
        'binary_validation_file_path': binary_validation_file_path_train_mode,
        'contain_zero_values': contain_zero_values,
        'address_near_zero_instability': address_near_zero_instability,
        'integer_conversion': integer_conversion,
        'seed': seed,
        'cell_type': cell_type,
        'without_stl_decomposition': without_stl_decomposition
    }

    # select the model type
    if model_type == "stacking":
        model_trainer = StackingModelTrainer(**model_kwargs)
    elif model_type == "seq2seq":
        if with_accumulated_error:
            model_trainer = Seq2SeqModelTrainerWithNonMovingWindowAccumulatedError(**model_kwargs)
        else:
            model_trainer = Seq2SeqModelTrainerWithNonMovingWindowUnaccumulatedError(**model_kwargs)
    elif model_type == "seq2seqwithdenselayer":
        if input_format == "non_moving_window":
            if with_accumulated_error:
                model_trainer = Seq2SeqModelTrainerWithDenseLayerNonMovingWindowAccumulatedError(**model_kwargs)
            else:
                model_trainer = Seq2SeqModelTrainerWithDenseLayerNonMovingWindowUnaccumulatedError(**model_kwargs)
        elif input_format == "moving_window":
            model_trainer = Seq2SeqModelTrainerWithDenseLayerMovingWindow(**model_kwargs)

    # read the initial hyperparamter configurations from the file
    hyperparameter_values_dic = read_initial_hyperparameter_values(initial_hyperparameter_values_file)
    optimized_configuration = smac()

    # persist the optimized configuration to a file
    persist_results(optimized_configuration, optimized_config_directory + '/' + model_identifier + '.txt')

    # get the validation errors for the best hyperparameter configs
    smape_error, smape_error_list = train_model(optimized_configuration)


    # write the final list of validation errors to a file
    validation_errors_file = model_training_configs.VALIDATION_ERRORS_DIRECTORY + model_identifier + ".csv"
    with open(validation_errors_file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(smape_error_list)

    print("Optimized configuration: {}".format(optimized_configuration))
    print("Optimized Value: {}\n".format(smape_error))

    # test the model
    for i in range(1, 11):
        args.seed = i
        testing(args, optimized_configuration)