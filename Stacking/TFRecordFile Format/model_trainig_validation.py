import numpy as np
import pandas as pd
import random

from bayes_opt import BayesianOptimization

import tensorflow as tf

# Input/Output Window sizes
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Training and Validation file paths.
binary_train_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/Binary Files/stl_12i15.tfrecords'
binary_validation_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/Binary Files/stl_12i15v.tfrecords'

# TODO: how to use the zip format in the binary file?

def L1Loss(z, t):
    loss = tf.reduce_mean(tf.abs(t - z))
    return loss

def parser(serialized_example):
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example,
        context_features=({
            "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
        }),
        sequence_features=({
            "input": tf.FixedLenSequenceFeature([INPUT_SIZE], dtype=tf.float32),
            "output": tf.FixedLenSequenceFeature([OUTPUT_SIZE], dtype=tf.float32),
            "metadata": tf.FixedLenSequenceFeature([OUTPUT_SIZE + 1], dtype=tf.float32)
        })
    )

    return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["output"], sequence_parsed["metadata"]

# Training the time series
def train_model(learningRate, lstmCellDimension, mbSize, maxEpochSize, maxNumOfEpochs, l2_regularization):

    tf.reset_default_graph()

    # declare the input and output placeholders
    input = tf.placeholder(dtype = tf.float32, shape = [None, None, INPUT_SIZE])
    label = tf.placeholder(dtype = tf.float32, shape = [None, None, OUTPUT_SIZE])
    sequence_lengths = tf.placeholder(dtype=tf.int64, shape=[None])

    # create the model architecture

    # RNN with the LSTM layer
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = int(lstmCellDimension), use_peepholes = LSTM_USE_PEEPHOLES)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = input, sequence_length = sequence_lengths, dtype = tf.float32)

    # connect the dense layer to the RNN
    dense_layer = tf.layers.dense(inputs = tf.convert_to_tensor(value = rnn_outputs, dtype = tf.float32), units = OUTPUT_SIZE, use_bias = BIAS)

    # error that should be minimized the training process
    error = L1Loss(dense_layer, label)

    # l2 regularization of the trainable model parameters
    l2_loss = 0.0
    for var in tf.trainable_variables() :
        l2_loss += tf.nn.l2_loss(var)

    l2_loss = tf.multiply(l2_regularization, tf.cast(l2_loss, tf.float64))

    total_loss = tf.cast(error, tf.float64) + l2_loss

    # create the adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate = learningRate).minimize(total_loss)

    # create the training and validation datasets from the tfrecord files
    training_dataset = tf.data.TFRecordDataset([binary_train_file_path])
    validation_dataset = tf.data.TFRecordDataset([binary_validation_file_path])

    # parse the records
    training_dataset = training_dataset.map(parser)  # TODO: optimize this more with the variables in the function
    validation_dataset = validation_dataset.map(parser)  # TODO: optimize this more with the variables in the function

    # define the expected shapes of data after padding
    padded_shapes = ([], [tf.Dimension(None), INPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE + 1])

    INFO_FREQ = 1
    sMAPE_final_list = []

    # setup variable initialization
    init_op = tf.global_variables_initializer()

    with tf.Session() as session :
        session.run(init_op)

        for iscan in range(int(maxNumOfEpochs)):
            sMAPE_epoch_list = []
            print("Epoch->", iscan)

            # randomly shuffle the time series within the dataset
            training_dataset.shuffle(int(mbSize))

            for epochsize in range(int(maxEpochSize)):
                sMAPE_list = []
                padded_training_data_batches = training_dataset.padded_batch(batch_size=10, padded_shapes=padded_shapes)

                training_data_batch_iterator = padded_training_data_batches.make_one_shot_iterator()
                next_training_data_batch = training_data_batch_iterator.get_next()


                while True:
                    try:
                        training_data_batch_value = session.run(next_training_data_batch)
                        session.run(optimizer,
                                    feed_dict={input: training_data_batch_value[1],
                                               label: training_data_batch_value[2],
                                               sequence_lengths: training_data_batch_value[0]})
                    except tf.errors.OutOfRangeError:
                        break

                if iscan % INFO_FREQ == 0:
                    # create a single batch from all the validation time series by padding the datasets to make the variable sequence lengths fixed
                    padded_validation_dataset = validation_dataset.padded_batch(batch_size = mbSize, padded_shapes = padded_shapes)

                    # get an iterator to the validation data
                    validation_data_iterator = padded_validation_dataset.make_one_shot_iterator()

                    while True:
                        try:
                            # access the validation data using the iterator
                            next_validation_data_batch = validation_data_iterator.get_next()

                            # get the batch of validation inputs
                            validation_data_batch_value = session.run(next_validation_data_batch)

                            # get the output of the network for the validation input data batch
                            validation_output = session.run(dense_layer, feed_dict={input: validation_data_batch_value[1],
                                                                                    sequence_lengths: validation_data_batch_value[0]
                                                                                    })

                            # calculate the smape for the validation data using vectorization

                            # convert the data to remove the preprocessing
                            last_indices = validation_data_batch_value[0] - 1
                            array_first_dimension = np.array(range(0, validation_data_batch_value[0].shape[0]))

                            true_seasonality_values = validation_data_batch_value[3][array_first_dimension, last_indices, 1:]
                            level_values = validation_data_batch_value[3][array_first_dimension, last_indices, 0]

                            last_validation_outputs = validation_output[array_first_dimension, last_indices]
                            converted_validation_output = true_seasonality_values + level_values[:, np.newaxis] + last_validation_outputs

                            actual_values = validation_data_batch_value[2][array_first_dimension, last_indices, :]
                            converted_actual_values = true_seasonality_values + level_values[:, np.newaxis] + actual_values

                            # calculate the smape
                            sMAPE = np.mean(np.abs(converted_validation_output - converted_actual_values) /
                                                (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
                            sMAPE_list.append(sMAPE)

                        except tf.errors.OutOfRangeError:
                            break
                sMAPEprint = np.mean(sMAPE_list)
                sMAPE_epoch_list.append(sMAPEprint)

            sMAPE_validation = np.mean(sMAPE_epoch_list)
            sMAPE_final_list.append(sMAPE_validation)
            # print("smape: ", sMAPE_validation)

        sMAPE_final = np.mean(sMAPE_final_list)
        max_value = 1 / (sMAPE_final)

    return max_value
    # return 0

if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)

    init_points = 2
    num_iter = 30
    time_series = 15
    mini_batch_size = 10

    # create_train_data()

    # using bayesian optimizer for hyperparameter optimization
    bayesian_optimization = BayesianOptimization(train_model, {'learningRate': (0.0001, 0.0008),
                                                     'lstmCellDimension': (10, 100),
                                                     'mbSize': (10, 30),
                                                     'maxEpochSize': (1, 1),
                                                     'maxNumOfEpochs': (3, 20),
                                                     'l2_regularization': (0.0001, 0.0008)
                                                     # 'gaussianNoise': (0.0001, 0.0008)
                                                        })

    bayesian_optimization.maximize(init_points = init_points, n_iter = num_iter)
    # print(bayesian_optimization.res['max'])
    # train_model(learningRate=0.0013262220421187676, lstmCellDimension=2, mbSize=mini_batch_size, maxEpochSize=1, maxNumOfEpochs=20,
    #             l2_regularization=0.00015753660121731034
                # , gaussianNoise=0.00023780395225712772
                # )

