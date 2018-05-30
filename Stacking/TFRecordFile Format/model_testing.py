import numpy as np
import pandas as pd
import random
import csv

import tensorflow as tf

# Input/Output Window size.
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Training and Validation file paths.
binary_train_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/Binary Files/stl_12i15.tfrecords'
binary_test_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/Binary Files/cif12test.tfrecords'
forecast_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/forecasts.txt'

def L1Loss(z, t):
    loss = tf.reduce_mean(tf.abs(t - z))
    return loss

def train_data_parser(serialized_example):
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

def test_data_parser(serialized_example):
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example,
        context_features=({
            "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
        }),
        sequence_features=({
            "input": tf.FixedLenSequenceFeature([INPUT_SIZE], dtype=tf.float32),
            "metadata": tf.FixedLenSequenceFeature([OUTPUT_SIZE + 1], dtype=tf.float32)
        })
    )

    return context_parsed["sequence_length"], sequence_parsed["input"], sequence_parsed["metadata"]

# Training the time series
def train_model():

    # optimized hyperparameters
    maxNumOfEpochs = 5.0381
    maxEpochSize = 1
    learningRate = 0.0007
    lstmCellDimension = 55.3103
    l2_regularization = 0.0001
    mbSize = 14.3145

    # reset the tensorflow graph
    tf.reset_default_graph()

    # declare the input and output placeholders
    input = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
    label = tf.placeholder(dtype=tf.float32, shape=[None, None, OUTPUT_SIZE])
    sequence_lengths = tf.placeholder(dtype=tf.int64, shape=[None])

    # create the model architecture

    # RNN with the LSTM layer
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(lstmCellDimension),
                                        use_peepholes=LSTM_USE_PEEPHOLES)  # TODO: self stabilization - not quite needed

    rnn_outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input, sequence_length=sequence_lengths,
                                            dtype=tf.float32)

    # connect the dense layer to the RNN
    dense_layer = tf.layers.dense(inputs=tf.convert_to_tensor(value=rnn_outputs, dtype=tf.float32), units=OUTPUT_SIZE,
                                  use_bias=BIAS)

    # error that should be minimized the training process
    error = L1Loss(dense_layer, label)

    # l2 regularization of the trainable model parameters
    l2_loss = 0.0
    for var in tf.trainable_variables():
        l2_loss += tf.nn.l2_loss(var)

    l2_loss = tf.multiply(l2_regularization, l2_loss)

    total_loss = error + l2_loss

    # create the adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate=learningRate).minimize(
        total_loss)  # TODO: gaussian_noise_injection_std_dev=gaussianNoise

    # create the Dataset objects for the training and test data
    training_dataset = tf.data.TFRecordDataset([binary_train_file_path])
    test_dataset = tf.data.TFRecordDataset([binary_test_file_path])

    # parse the records
    training_dataset = training_dataset.map(train_data_parser)  # TODO: optimize this more with the variables in the function
    test_dataset = test_dataset.map(test_data_parser)  # TODO: optimize this more with the variables in the function

    # setup variable initialization
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        for iscan in range(int(maxNumOfEpochs)):
            print("Epoch->", iscan)

            # randomly shuffle the time series within the dataset
            training_dataset.shuffle(int(mbSize))

            for epochsize in range(int(maxEpochSize)):

                # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
                padded_training_data_batches = training_dataset.padded_batch(batch_size = int(mbSize),
                                      padded_shapes = ([], [tf.Dimension(None), INPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE + 1]))

                # get an iterator to the batches
                training_data_batch_iterator = padded_training_data_batches.make_one_shot_iterator()

                # access each batch using the iterator
                next_training_data_batch = training_data_batch_iterator.get_next()

                while True:
                    try:
                        next_training_batch_value = session.run(next_training_data_batch)

                        # model training
                        session.run(optimizer,
                                    feed_dict={input: next_training_batch_value[1],
                                               label: next_training_batch_value[2],
                                               sequence_lengths: next_training_batch_value[0]})
                    except tf.errors.OutOfRangeError:
                        break

        # applying the model to the test data

        # create a single batch from all the test time series by padding the datasets to make the variable sequence lengths fixed
        padded_test_input_data = test_dataset.padded_batch(batch_size=int(mbSize), padded_shapes = ([], [tf.Dimension(None), INPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE + 1]))

        # get an iterator to the test input data batch
        test_input_iterator = padded_test_input_data.make_one_shot_iterator()

        list_of_forecasts = []
        while True:
            try:
                # access the test input batch using the iterator
                test_input_data_batch = test_input_iterator.get_next()

                # get the batch of test inputs
                test_input_batch_value = session.run(test_input_data_batch)

                # get the output of the network for the test input data batch
                test_output = session.run(dense_layer,
                                          feed_dict={input: test_input_batch_value[1],
                                                     sequence_lengths: test_input_batch_value[0]})

                last_output_index = test_input_batch_value[0] - 1
                array_first_dimension = np.array(range(0, test_input_batch_value[0].shape[0]))
                forecasts = test_output[array_first_dimension, last_output_index]
                list_of_forecasts.extend(forecasts.tolist())

            except tf.errors.OutOfRangeError:
                break
        tf.truncated_normal

        return list_of_forecasts


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)

    list_of_forecasts = train_model()

    # # Writes the last test output(i.e Forecast) of each time series to a file
    # for time_series in range(len(test_output)):
    #     last_output_index = output_sequence_lengths[time_series] - 1
    #     forecast = test_output[time_series][last_output_index]
    #     list_of_forecasts.append(forecast)

    with open(forecast_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_forecasts)
