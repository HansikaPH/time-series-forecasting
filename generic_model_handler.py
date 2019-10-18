import numpy as np
import argparse
import csv


from utility_scripts.persist_optimized_config_results import persist_results
# from utility_scripts.hyperparameter_scripts.hyperparameter_config_reader import read_optimal_hyperparameter_values
from utility_scripts.hyperparameter_scripts.hyperparameter_config_reader import read_initial_hyperparameter_values
from configs.global_configs import hyperparameter_tuning_configs
from configs.global_configs import model_testing_configs
from utility_scripts.invoke_r_final_evaluation import invoke_r_script

# import SMAC utilities
# import the config space and the different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

## import the different model architectures

# stacking model
from rnn_architectures.stacking_model import StackingModel

# # seq2seq model with decoder
# from rnn_architectures.seq2seq_model.with_decoder.non_moving_window.unaccumulated_error.seq2seq_model_trainer import \
#     Seq2SeqModelTrainer as Seq2SeqModelTrainerWithNonMovingWindowUnaccumulatedError
# from rnn_architectures.seq2seq_model.with_decoder.non_moving_window.accumulated_error.seq2seq_model_trainer import \
#     Seq2SeqModelTrainer as Seq2SeqModelTrainerWithNonMovingWindowAccumulatedError
#
# # seq2seq model with dense layer
# from rnn_architectures.seq2seq_model.with_dense_layer.non_moving_window.unaccumulated_error.seq2seq_model_trainer import \
#     Seq2SeqModelTrainerWithDenseLayer as Seq2SeqModelTrainerWithDenseLayerNonMovingWindowUnaccumulatedError
# from rnn_architectures.seq2seq_model.with_dense_layer.non_moving_window.accumulated_error.seq2seq_model_trainer import \
#     Seq2SeqModelTrainerWithDenseLayer as Seq2SeqModelTrainerWithDenseLayerNonMovingWindowAccumulatedError
# from rnn_architectures.seq2seq_model.with_dense_layer.moving_window.unaccumulated_error.seq2seq_model_trainer import \
#     Seq2SeqModelTrainerWithDenseLayer as Seq2SeqModelTrainerWithDenseLayerMovingWindow

LSTM_USE_PEEPHOLES = True
BIAS = False

# final execution with the optimized config
def train_model(configs):
    print(configs)

    hyperparameter_values = {
        "num_hidden_layers": configs["num_hidden_layers"],
        "cell_dimension": configs["cell_dimension"],
        "minibatch_size": configs["minibatch_size"],
        "max_epoch_size": configs["max_epoch_size"],
        "max_num_epochs": configs["max_num_epochs"],
        "l2_regularization": configs["l2_regularization"],
        "gaussian_noise_stdev": configs["gaussian_noise_stdev"],
        "random_normal_initializer_stdev": configs["random_normal_initializer_stdev"],
    }

    if optimizer != "cocob":
        hyperparameter_values["initial_learning_rate"] = configs["initial_learning_rate"]

    error = model.tune_hyperparameters(**hyperparameter_values)

    print(model_identifier)
    print(error)
    return error.item()

def smac():
    # Build Configuration Space which defines all parameters and their ranges
    configuration_space = ConfigurationSpace()

    initial_learning_rate = UniformFloatHyperparameter("initial_learning_rate", hyperparameter_values_dic['initial_learning_rate'][0],
                                                  hyperparameter_values_dic['initial_learning_rate'][1],
                                                  default_value=hyperparameter_values_dic['initial_learning_rate'][0])
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
            [initial_learning_rate, cell_dimension, minibatch_size, max_epoch_size,
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
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(seed), tae_runner=train_model)

    incumbent = smac.optimize()
    return incumbent.get_dictionary()


if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser("Train different forecasting models")
    argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
    argument_parser.add_argument('--contain_zero_values', required=False,
                                 help='Whether the dataset contains zero values(0/1). Default is 0')
    argument_parser.add_argument('--address_near_zero_instability', required=False,
                                 help='Whether to use a custom SMAPE function to address near zero instability(0/1). Default is 0')
    argument_parser.add_argument('--integer_conversion', required=False,
                                 help='Whether to convert the final forecasts to integers(0/1). Default is 0')
    argument_parser.add_argument('--no_of_series', required=True,
                                 help='The number of series in the dataset')
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
    argument_parser.add_argument('--optimizer', required=False, help='The type of the optimizer(cocob/adam/adagrad...). Default is cocob')
    argument_parser.add_argument('--model_type', required=False,
                                 help='The type of the model(stacking/seq2seq/seq2seqwithdenselayer). Default is stacking')
    argument_parser.add_argument('--input_format', required=False, help='Input format(moving_window/non_moving_window). Default is moving_window')
    argument_parser.add_argument('--without_stl_decomposition', required=False,
                                 help='Whether not to use stl decomposition(0/1). Default is 0')
    argument_parser.add_argument('--with_accumulated_error', required=False,
                                 help='Whether not to use accumulated error when calculating the loss(0/1). Default is 0')

    # parse the user arguments
    args = argument_parser.parse_args()

    # arguments with no default values
    dataset_name = args.dataset_name
    no_of_series = int(args.no_of_series)
    initial_hyperparameter_values_file = args.initial_hyperparameter_values_file
    binary_train_file_path_train_mode = args.binary_train_file_train_mode
    binary_validation_file_path_train_mode = args.binary_valid_file_train_mode
    binary_test_file_path_test_mode = args.binary_test_file_test_mode
    output_size = int(args.forecast_horizon)
    txt_test_file_path = args.txt_test_file
    actual_results_file_path = args.actual_results_file
    original_data_file_path = args.original_data_file
    seasonality_period = int(args.seasonality_period)
    seed = 1234

    # arguments with default values
    if args.contain_zero_values:
        contain_zero_values = bool(int(args.contain_zero_values))
    else:
        contain_zero_values = False

    if args.input_size:
        input_size = int(args.input_size)
    else:
        input_size = 0

    if args.optimizer:
        optimizer = args.optimizer
    else:
        optimizer = "cocob"

    if args.model_type:
        model_type = args.model_type
    else:
        model_type = "stacking"

    if args.input_format:
        input_format = args.input_format
    else:
        input_format = "moving_window"

    if args.without_stl_decomposition:
        without_stl_decomposition = bool(int(args.without_stl_decomposition))
    else:
        without_stl_decomposition = False

    if args.cell_type:
        cell_type = args.cell_type
    else:
        cell_type = "LSTM"

    if args.address_near_zero_instability:
        address_near_zero_instability = bool(int(args.address_near_zero_instability))
    else:
        address_near_zero_instability = False

    if args.integer_conversion:
        integer_conversion = bool(int(args.integer_conversion))
    else:
        integer_conversion = False

    if args.with_accumulated_error:
        with_accumulated_error = bool(int(args.with_accumulated_error))
    else:
        with_accumulated_error = False

    if without_stl_decomposition:
        stl_decomposition_identifier = "without_stl_decomposition"
    else:
        stl_decomposition_identifier = "with_stl_decomposition"

    if with_accumulated_error:
        accumulated_error_identifier = "with_accumulated_error"
    else:
        accumulated_error_identifier = "without_accumulated_error"


    model_identifier = dataset_name + "_" + model_type + "_" + input_format + "_" + cell_type + "cell" + "_" +  optimizer + "_" + \
                       stl_decomposition_identifier + "_" + accumulated_error_identifier
    print("Model Training Started for {}".format(model_identifier))


    # define the key word arguments for the different model types
    model_kwargs = {
        'use_bias': BIAS,
        'use_peepholes': LSTM_USE_PEEPHOLES,
        'input_size': input_size,
        'output_size': output_size,
        'optimizer': optimizer,
        'no_of_series': no_of_series,
        'binary_train_file_path': binary_train_file_path_train_mode,
        'binary_test_file_path': binary_test_file_path_test_mode,
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
        model = StackingModel(**model_kwargs)
    # elif model_type == "seq2seq":
    #     if with_accumulated_error:
    #         model_trainer = Seq2SeqModelTrainerWithNonMovingWindowAccumulatedError(**model_kwargs)
    #     else:
    #         model_trainer = Seq2SeqModelTrainerWithNonMovingWindowUnaccumulatedError(**model_kwargs)
    # elif model_type == "seq2seqwithdenselayer":
    #     if input_format == "non_moving_window":
    #         if with_accumulated_error:
    #             model_trainer = Seq2SeqModelTrainerWithDenseLayerNonMovingWindowAccumulatedError(**model_kwargs)
    #         else:
    #             model_trainer = Seq2SeqModelTrainerWithDenseLayerNonMovingWindowUnaccumulatedError(**model_kwargs)
    #     elif input_format == "moving_window":
    #         model_trainer = Seq2SeqModelTrainerWithDenseLayerMovingWindow(**model_kwargs)

    # read the initial hyperparamter configurations from the file
    hyperparameter_values_dic = read_initial_hyperparameter_values(initial_hyperparameter_values_file)
    optimized_configuration = smac()
    print(optimized_configuration)

    # persist the optimized configuration to a file
    persist_results(optimized_configuration, hyperparameter_tuning_configs.OPTIMIZED_CONFIG_DIRECTORY + '/' + model_identifier + '.txt')

    # optimized_configuration = read_optimal_hyperparameter_values("results/optimized_configurations/" + model_identifier + ".txt")

    for seed in range(1, 11):
        forecasts = model.test_model(optimized_configuration, seed)

        model_identifier_extended = model_identifier + "_" + str(seed)
        rnn_forecasts_file_path = model_testing_configs.RNN_FORECASTS_DIRECTORY + model_identifier_extended + '.txt'

        with open(rnn_forecasts_file_path, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(forecasts)

        # invoke the final evaluation R script
        error_file_name = model_identifier_extended + '.txt'

        invoke_r_script((rnn_forecasts_file_path, error_file_name, txt_test_file_path,
                         actual_results_file_path, original_data_file_path, str(input_size), str(output_size),
                         str(int(contain_zero_values)), str(int(address_near_zero_instability)),
                         str(int(integer_conversion)), str(int(seasonality_period)),
                         str(int(without_stl_decomposition))), True)

