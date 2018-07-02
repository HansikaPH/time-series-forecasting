import csv
import tensorflow as tf
import argparse

# import the different model types
from  stacking_model.stacking_model_trainer import StackingModelTrainer
from stacking_model.stacking_model_tester import StackingModelTester
# from seq2seq_model_trainer import Seq2SeqModelTrainer
from attention_model.attention_model_trainer import AttentionModelTrainer
from attention_model.attention_model_tester import AttentionModelTester

# import the cocob optimizer
from external_packages import cocob_optimizer

LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Input/Output Window sizes
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# Training and Validation file paths.
binary_train_file_path = 'datasets/CIF_2016/binary_files/stl_12i15.tfrecords'
binary_test_file_path = 'datasets/CIF_2016/binary_files/cif12test.tfrecords'

# Directory to save the forecasts
forecasts_directory = 'forecasts/'

# function to create the optimizer
def adagrad_optimizer_fn(total_loss):
    return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)

def cocob_optimizer_fn(total_loss):
    return cocob_optimizer.COCOB().minimize(loss=total_loss)

if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser("Test different forecasting models")
    argument_parser.add_argument('--optimizer', required = True, help = 'The type of the optimizer(cocob/adam/adagrad...)')
    argument_parser.add_argument('--hyperparameter_tuning', required=True,
                                 help='The method for hyperparameter tuning(bayesian/smac)')
    argument_parser.add_argument('--model_type', required=True, help='The type of the model(stacking/seq2seq/attention)')

    # parse the user arguments
    args = argument_parser.parse_args()

    optimizer = args.optimizer
    hyperparameter_tuning = args.hyperparameter_tuning
    model_type = args.model_type

    # select the optimizer
    if optimizer == "cocob":
        optimizer_fn = cocob_optimizer_fn
    elif optimizer == "adagrad":
        optimizer_fn = adagrad_optimizer_fn

    # select the model type
    if model_type == "stacking":
        model_class = StackingModelTester
    elif model_type == "attention":
        model_class = AttentionModelTester

    learning_rate = 0.0003426256821607555
    num_hidden_layers = 4.2530405625473344
    max_num_epochs = 3
    max_epoch_size = 3
    lstm_cell_dimension = 56.079257770813335
    l2_regularization = 0.00079999991895750083
    minibatch_size = 30
    gaussian_noise_stdev = 0.00079999991895750083

    model_tester = model_class(use_bias=BIAS,
                                use_peepholes=LSTM_USE_PEEPHOLES,
                                input_size=INPUT_SIZE,
                                output_size=OUTPUT_SIZE,
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
    forecast_file_path = forecasts_directory + model_type + '_' + hyperparameter_tuning + '_' + optimizer + '.txt'
    with open(forecast_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_forecasts)


