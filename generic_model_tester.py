import csv
import tensorflow as tf

# import the different model types

# stacking model
from rnn_architectures.stacking_model.stacking_model_tester import \
    StackingModelTester as StackingModelTester

# seq2seq model with decoder
from rnn_architectures.seq2seq_model.with_decoder.non_moving_window.unaccumulated_error.seq2seq_model_tester import \
    Seq2SeqModelTester as Seq2SeqModelTesterWithNonMovingWindowUnaccumulatedError

# seq2seq model with dense layer
from rnn_architectures.seq2seq_model.with_dense_layer.non_moving_window.unaccumulated_error.seq2seq_model_tester import \
    Seq2SeqModelTesterWithDenseLayer as Seq2SeqModelTesterWithDenseLayerNonMovingWindowUnaccumulatedError
from rnn_architectures.seq2seq_model.with_dense_layer.moving_window.unaccumulated_error.seq2seq_model_tester import \
    Seq2SeqModelTesterWithDenseLayer as Seq2SeqModelTesterWithDenseLayerMovingWindow


# import the cocob optimizer
from external_packages import cocob_optimizer
from utility_scripts.invoke_r_final_evaluation import invoke_r_script

from configs.global_configs import model_testing_configs

LSTM_USE_PEEPHOLES = True
BIAS = False

learning_rate = 0.0


# function to create the optimizer
def adagrad_optimizer_fn(total_loss):
    return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)


def adam_optimizer_fn(total_loss):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)


def cocob_optimizer_fn(total_loss):
    return cocob_optimizer.COCOB().minimize(loss=total_loss)


def testing(args, config_dictionary):
    # to make the random number choices reproducible

    global learning_rate

    dataset_name = args.dataset_name
    contain_zero_values = int(args.contain_zero_values)
    binary_train_file_path_test_mode = args.binary_train_file_test_mode
    binary_test_file_path_test_mode = args.binary_test_file_test_mode
    txt_test_file_path = args.txt_test_file
    actual_results_file_path = args.actual_results_file
    original_data_file_path = args.original_data_file

    if (args.input_size):
        input_size = int(args.input_size)
    else:
        input_size = 0
    output_size = int(args.forecast_horizon)
    seasonality_period = int(args.seasonality_period)
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

    if not with_truncated_backpropagation:
        tbptt_identifier = "without_truncated_backpropagation"
    else:
        tbptt_identifier = "with_truncated_backpropagation"

    if not without_stl_decomposition:
        stl_decomposition_identifier = "with_stl_decomposition"
    else:
        stl_decomposition_identifier = "without_stl_decomposition"

    if with_accumulated_error:
        accumulated_error_identifier = "with_accumulated_error"
    else:
        accumulated_error_identifier = "without_accumulated_error"

    model_identifier = dataset_name + "_" + model_type + "_" + cell_type + "cell" + "_" + input_format + "_" + stl_decomposition_identifier + "_" + hyperparameter_tuning + "_" + optimizer + "_" + tbptt_identifier + "_" + accumulated_error_identifier + "_" + str(
        seed)
    print("Model Testing Started for {}".format(model_identifier))
    print(config_dictionary)

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
        'binary_train_file_path': binary_train_file_path_test_mode,
        'binary_test_file_path': binary_test_file_path_test_mode,
        'seed': seed,
        'cell_type': cell_type,
        'without_stl_decomposition': without_stl_decomposition
    }

    # select the model type
    if model_type == "stacking":
        model_tester = StackingModelTester(**model_kwargs)
    elif model_type == "seq2seq":
        model_tester = Seq2SeqModelTesterWithNonMovingWindowUnaccumulatedError(**model_kwargs)
    elif model_type == "seq2seqwithdenselayer":
        if input_format == "non_moving_window":
            model_tester = Seq2SeqModelTesterWithDenseLayerNonMovingWindowUnaccumulatedError(**model_kwargs)
        elif input_format == "moving_window":
            model_tester = Seq2SeqModelTesterWithDenseLayerMovingWindow(**model_kwargs)

    if 'rate_of_learning' in config_dictionary:
        learning_rate = config_dictionary['rate_of_learning']
    num_hidden_layers = config_dictionary['num_hidden_layers']
    max_num_epochs = config_dictionary['max_num_epochs']
    max_epoch_size = config_dictionary['max_epoch_size']
    cell_dimension = config_dictionary['cell_dimension']
    l2_regularization = config_dictionary['l2_regularization']
    minibatch_size = config_dictionary['minibatch_size']
    gaussian_noise_stdev = config_dictionary['gaussian_noise_stdev']
    random_normal_initializer_stdev = config_dictionary['random_normal_initializer_stdev']

    list_of_forecasts = model_tester.test_model(num_hidden_layers=int(round(num_hidden_layers)),
                                                cell_dimension=int(round(cell_dimension)),
                                                minibatch_size=int(round(minibatch_size)),
                                                max_epoch_size=int(round(max_epoch_size)),
                                                max_num_epochs=int(round(max_num_epochs)),
                                                l2_regularization=l2_regularization,
                                                gaussian_noise_stdev=gaussian_noise_stdev,
                                                random_normal_initializer_stdev=random_normal_initializer_stdev,
                                                optimizer_fn=optimizer_fn)

    # write the forecasting results to a file
    rnn_forecasts_file_path = model_testing_configs.RNN_FORECASTS_DIRECTORY + model_identifier + '.txt'

    with open(rnn_forecasts_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_forecasts)

    # invoke the final evaluation R script
    error_file_name = model_identifier + '.txt'

    if input_format == "moving_window":
            invoke_r_script((rnn_forecasts_file_path, error_file_name, txt_test_file_path,
                             actual_results_file_path, original_data_file_path, str(input_size), str(output_size),
                             str(contain_zero_values), str(int(address_near_zero_instability)),
                             str(int(integer_conversion)), str(int(seasonality_period)), str(int(without_stl_decomposition))), True)

    else:
            invoke_r_script((rnn_forecasts_file_path, error_file_name, txt_test_file_path,
                             actual_results_file_path, original_data_file_path, str(output_size), str(contain_zero_values), str(int(address_near_zero_instability)), str(int(integer_conversion)), str(int(seasonality_period)), str(int(without_stl_decomposition))), False)

