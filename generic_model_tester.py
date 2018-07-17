import csv
import tensorflow as tf
import argparse

# import the different model types
from stacking_model.stacking_model_tester import StackingModelTester
from seq2seq_model.seq2seq_model_tester import Seq2SeqModelTester
from attention_model.attention_model_tester import AttentionModelTester

# import the cocob optimizer
from external_packages import cocob_optimizer
from invoke_r_final_evaluation import invoke_r_script

LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Directory to save the forecasts
forecasts_directory = 'results/forecasts/'
learning_rate = 0.0

# function to create the optimizer
def adagrad_optimizer_fn(total_loss):
    return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)

def cocob_optimizer_fn(total_loss):
    return cocob_optimizer.COCOB().minimize(loss=total_loss)

def testing(args, config_dictionary):

    # argument_parser = argparse.ArgumentParser("Test different forecasting models on different datasets")
    # argument_parser.add_argument('--binary_train_file', required=True, help='The tfrecords file for train dataset')
    # argument_parser.add_argument('--binary_test_file', required=True, help='The tfrecords file for test dataset')
    # argument_parser.add_argument('--input_size', required=True, help='The input size of the dataset')
    # argument_parser.add_argument('--forecast_horizon', required=True, help='The forecast horizon of the dataset')
    # argument_parser.add_argument('--optimizer', required = True, help = 'The type of the optimizer(cocob/adam/adagrad...)')
    # argument_parser.add_argument('--hyperparameter_tuning', required=True,
    #                              help='The method for hyperparameter tuning(bayesian/smac)')
    # argument_parser.add_argument('--model_type', required=True, help='The type of the model(stacking/seq2seq/attention)')
    #
    # # parse the user arguments
    # args = argument_parser.parse_args()
    global learning_rate

    dataset_name = args.dataset_name
    contain_zero_values = args.contain_zero_values
    binary_train_file_path = args.binary_train_file
    binary_test_file_path = args.binary_test_file
    txt_test_file_path = args.txt_test_file
    actual_results_file_path = args.actual_results_file
    input_size = int(args.input_size)
    output_size = int(args.forecast_horizon)
    optimizer = args.optimizer
    hyperparameter_tuning = args.hyperparameter_tuning
    model_type = args.model_type

    print("Model Testing Started for {}_{}_{}_{}".format(dataset_name, model_type, hyperparameter_tuning, optimizer))

    # select the optimizer
    if optimizer == "cocob":
        optimizer_fn = cocob_optimizer_fn
    elif optimizer == "adagrad":
        optimizer_fn = adagrad_optimizer_fn

    # select the model type
    if model_type == "stacking":
        model_class = StackingModelTester
    elif model_type == "seq2seq":
        model_class = Seq2SeqModelTester
    elif model_type == "attention":
        model_class = AttentionModelTester

    if 'learning_rate' in config_dictionary:
        learning_rate = config_dictionary['learning_rate']
    num_hidden_layers = config_dictionary['num_hidden_layers']
    max_num_epochs = config_dictionary['max_num_epochs']
    max_epoch_size = config_dictionary['max_epoch_size']
    lstm_cell_dimension = config_dictionary['lstm_cell_dimension']
    l2_regularization = config_dictionary['l2_regularization']
    minibatch_size = config_dictionary['minibatch_size']
    gaussian_noise_stdev = config_dictionary['gaussian_noise_stdev']

    model_tester = model_class(use_bias=BIAS,
                                use_peepholes=LSTM_USE_PEEPHOLES,
                                input_size=input_size,
                                output_size=output_size,
                                binary_train_file_path=binary_train_file_path,
                                binary_test_file_path=binary_test_file_path)

    list_of_forecasts = model_tester.test_model(num_hidden_layers = int(round(num_hidden_layers)),
                                      lstm_cell_dimension = int(round(lstm_cell_dimension)),
                                      minibatch_size = int(round(minibatch_size)),
                                      max_epoch_size = int(round(max_epoch_size)),
                                      max_num_epochs = int(round(max_num_epochs)),
                                      l2_regularization = l2_regularization,
                                      gaussian_noise_stdev = gaussian_noise_stdev,
                                      optimizer_fn = optimizer_fn)

    # write the forecasting results to a file
    forecast_file_path = forecasts_directory + dataset_name + '_' + model_type + '_' + hyperparameter_tuning + '_' + optimizer + '.txt'
    with open(forecast_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_forecasts)

    # invoke the final evaluation R script
    error_file_name = dataset_name + '_' + model_type + '_' + hyperparameter_tuning + '_' + optimizer + '.txt'
    invoke_r_script((forecast_file_path, error_file_name, txt_test_file_path, actual_results_file_path, str(input_size), str(output_size), contain_zero_values))

