import sys
import csv

from generic_model_tester import test_model
import tensorflow as tf

# Input/Output Window size.
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# function to create the model
def create_model(input, sequence_lengths, lstm_cell_dimension, use_peepholes, num_hidden_layers):

    # RNN with the LSTM layer
    def lstm_cell():
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = lstm_cell_dimension, use_peepholes=use_peepholes)
        return lstm_cell

    multi_layered_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell() for _ in range(num_hidden_layers)])
    rnn_outputs, states = tf.nn.dynamic_rnn(cell=multi_layered_cell, inputs=input, sequence_length=sequence_lengths,
                                            dtype=tf.float32)

    # connect the dense layer to the RNN
    dense_layer = tf.layers.dense(inputs=tf.convert_to_tensor(value=rnn_outputs, dtype=tf.float32), units=OUTPUT_SIZE,
                                  use_bias=BIAS)
    return dense_layer

if __name__ == '__main__':

    # optimized hyperparameters
    no_hidden_layers = 5
    max_no_of_epochs = 20
    max_epoch_size = 3
    learning_rate = 0.0006229434571986981
    lstm_cell_dimension = 80
    l2_regularization = 0.0005950321895642069
    minibatch_size = 10.367370648328363
    gaussian_noise_std = 0.0004274864418226984


    list_of_forecasts = test_model(no_hidden_layers, max_no_of_epochs, max_epoch_size, learning_rate, lstm_cell_dimension, l2_regularization, minibatch_size, gaussian_noise_std, create_model)
    forecast_file_path = sys.argv[1]

    with open(forecast_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_forecasts)
