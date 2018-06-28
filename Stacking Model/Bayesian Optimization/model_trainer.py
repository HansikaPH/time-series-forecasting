import numpy as np
from bayes_opt import BayesianOptimization

import tensorflow as tf

from generic_model_trainer import train_model

LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Input/Output Window sizes
INPUT_SIZE = 15
OUTPUT_SIZE = 12

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

# Training the time series
def common_train_model(learning_rate, no_hidden_layers, lstm_cell_dimension, minibatch_size, max_epoch_size, max_num_of_epochs, l2_regularization, gaussian_noise_stdev):

    error = train_model(learning_rate, no_hidden_layers, lstm_cell_dimension, minibatch_size, max_epoch_size, max_num_of_epochs, l2_regularization, gaussian_noise_stdev, create_model)
    return -1 * error

if __name__ == '__main__':

    init_points = 2
    num_iter = 30

    # using bayesian optimizer for hyperparameter optimization
    bayesian_optimization = BayesianOptimization(common_train_model, {'learning_rate': (0.0001, 0.0008),
                                                               'no_hidden_layers': (1, 5),
                                                                'lstm_cell_dimension': (50, 100),
                                                                'minibatch_size': (10, 30),
                                                                'max_epoch_size': (1, 3),
                                                                'max_num_of_epochs': (3, 20),
                                                                      'l2_regularization': (0.0001, 0.0008),
                                                                      'gaussian_noise_stdev': (0.0001, 0.0008)
                                                                      })

    bayesian_optimization.maximize(init_points = init_points, n_iter = num_iter)

