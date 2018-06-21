import sys

import numpy as np
import csv

import tensorflow as tf
from tensorflow.python.layers.core import Dense

# import the cocob optimizer
sys.path.insert(0, '../External Packages/cocob_optimizer/')
import cocob_optimizer

# Input/Output Window size.
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Training and Validation file paths.
binary_train_file_path = '../DataSets/CIF 2016/Binary Files/stl_12i15v.tfrecords'
binary_test_file_path = '../DataSets/CIF 2016/Binary Files/cif12test.tfrecords'

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
def train_model():

    # optimized hyperparameters
    max_no_of_epochs = 7
    max_epoch_size = 3
    lstm_cell_dimension = 24
    l2_regularization = 0.0005707363209984838
    minibatch_size = 27
    gaussian_noise_stdev = 0.00015699946725595746

    # reset the tensorflow graph
    tf.reset_default_graph()

    tf.set_random_seed(1)

    # declare the input and output placeholders

    # adding noise to the input
    input = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
    noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
    input = input + noise
    target = tf.placeholder(dtype=tf.float32, shape=[None, None, OUTPUT_SIZE])

    # placeholder for the sequence lengths
    sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

    # create the model architecture

    # building the encoder network
    encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(lstm_cell_dimension), use_peepholes=LSTM_USE_PEEPHOLES)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=input, sequence_length=sequence_length,
                                                       dtype=tf.float32)

    # decoder cell of the decoder network
    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_dimension, use_peepholes=LSTM_USE_PEEPHOLES)

    # the final projection layer to convert the output to the desired dimension
    dense_layer = Dense(units=OUTPUT_SIZE, use_bias=BIAS)

    # building the decoder network for training
    with tf.variable_scope('decode'):
        helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=target, sequence_length=sequence_length,
                                                                  sampling_probability=0.0)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=encoder_state,
                                                  output_layer=dense_layer)

        # perform the decoding
        training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

    # building the decoder network for inference
    with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
        helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=target, sequence_length=sequence_length,
                                                                  sampling_probability=1.0)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                  initial_state=encoder_state, output_layer=dense_layer)

        # perform the decoding
        inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

    # error that should be minimized in the training process
    error = l1_loss(training_decoder_outputs[0], target)

    # l2 regularization of the trainable model parameters
    l2_loss = 0.0
    for var in tf.trainable_variables():
        l2_loss += tf.nn.l2_loss(var)

    l2_loss = tf.multiply(l2_regularization, l2_loss)

    total_loss = error + l2_loss

    # create the adagrad optimizer
    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(
    #     total_loss)
    # create the cocob optimizer
    optimizer = cocob_optimizer.COCOB().minimize(loss=total_loss)

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
                                               target: next_training_batch_value[2],
                                               sequence_length: next_training_batch_value[0].astype(np.int32)})
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

                # shape for the target data
                target_data_shape = [np.shape(test_input_batch_value[1])[0], np.shape(test_input_batch_value[1])[1], OUTPUT_SIZE]

                # get the output of the network for the test input data batch
                test_output = session.run(inference_decoder_outputs[0],
                                          feed_dict={input: test_input_batch_value[1],
                                                     target: np.zeros(shape = target_data_shape),
                                                     sequence_length: test_input_batch_value[0]})

                last_output_index = test_input_batch_value[0] - 1
                array_first_dimension = np.array(range(0, test_input_batch_value[0].shape[0]))
                forecasts = test_output[array_first_dimension, last_output_index]
                list_of_forecasts.extend(forecasts.tolist())

            except tf.errors.OutOfRangeError:
                break

        return list_of_forecasts


if __name__ == '__main__':
    list_of_forecasts = train_model()
    forecast_file_path = sys.argv[1]

    with open(forecast_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_forecasts)
