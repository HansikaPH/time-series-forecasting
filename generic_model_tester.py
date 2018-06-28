import sys

import numpy as np
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
binary_train_file_path = '../../DataSets/CIF 2016/Binary Files/stl_12i15v.tfrecords'
binary_test_file_path = '../../DataSets/CIF 2016/Binary Files/cif12test.tfrecords'

def l1_loss(z, t):
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
def test_model(no_hidden_layers, max_no_of_epochs, max_epoch_size, learning_rate, lstm_cell_dimension, l2_regularization, minibatch_size, gaussian_noise_std, create_model):

    # reset the tensorflow graph
    tf.reset_default_graph()

    tf.set_random_seed(1)

    # declare the input and output placeholders
    input = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])

    # adding the gassian noise to the input layer
    noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_std, dtype=tf.float32)
    input = input + noise

    true_output = tf.placeholder(dtype=tf.float32, shape=[None, None, OUTPUT_SIZE])
    sequence_lengths = tf.placeholder(dtype=tf.int64, shape=[None])

    # create the model architecture

    prediction_output = create_model(input = input, sequence_lengths = sequence_lengths, lstm_cell_dimension = int(lstm_cell_dimension), use_peepholes = LSTM_USE_PEEPHOLES, num_hidden_layers = int(no_hidden_layers))

    # error that should be minimized in the training process
    error = l1_loss(prediction_output, true_output)

    # l2 regularization of the trainable model parameters
    l2_loss = 0.0
    for var in tf.trainable_variables():
        l2_loss += tf.nn.l2_loss(var)

    l2_loss = tf.multiply(l2_regularization, l2_loss)

    total_loss = error + l2_loss

    # create the adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(
        total_loss)

    # create the Dataset objects for the training and test data
    training_dataset = tf.data.TFRecordDataset(filenames = [binary_train_file_path], compression_type = "ZLIB")
    test_dataset = tf.data.TFRecordDataset([binary_test_file_path], compression_type = "ZLIB")

    # parse the records
    training_dataset = training_dataset.map(train_data_parser)
    test_dataset = test_dataset.map(test_data_parser)

    # setup variable initialization
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        for epoch in range(int(max_no_of_epochs)):
            print("Epoch->", epoch)

            # randomly shuffle the time series within the dataset
            training_dataset.shuffle(int(minibatch_size))

            for epochsize in range(int(max_epoch_size)):

                # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
                padded_training_data_batches = training_dataset.padded_batch(batch_size = int(minibatch_size),
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
                                               true_output: next_training_batch_value[2],
                                               sequence_lengths: next_training_batch_value[0]})
                    except tf.errors.OutOfRangeError:
                        break

        # applying the model to the test data

        # create a single batch from all the test time series by padding the datasets to make the variable sequence lengths fixed
        padded_test_input_data = test_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes = ([], [tf.Dimension(None), INPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE + 1]))

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
                test_output = session.run(prediction_output,
                                          feed_dict={input: test_input_batch_value[1],
                                                     sequence_lengths: test_input_batch_value[0]})

                last_output_index = test_input_batch_value[0] - 1
                array_first_dimension = np.array(range(0, test_input_batch_value[0].shape[0]))
                forecasts = test_output[array_first_dimension, last_output_index]
                list_of_forecasts.extend(forecasts.tolist())

            except tf.errors.OutOfRangeError:
                break

        return list_of_forecasts