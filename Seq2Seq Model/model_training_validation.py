import sys

import numpy as np

# import the config space and the different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

#import SMAC utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import tensorflow as tf
from tensorflow.python.layers.core import Dense

# import the cocob optimizer
sys.path.insert(0, '../External Packages/cocob_optimizer/')
import cocob_optimizer

# Input/Output Window sizes
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Training and Validation file paths.
binary_train_file_path = '../DataSets/CIF 2016/Binary Files/stl_12i15.tfrecords'
binary_validation_file_path = '../DataSets/CIF 2016/Binary Files/stl_12i15v.tfrecords'

# TODO: lstm cell dimension for encoder and decoder
# TODO: why decoder_outputs[0]?
# TODO: integrate the attention mechanism
# TODO: see if the model architecture is reused

def l1_loss(z, t):
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

def gaussian_noise(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

# Training the time series
def train_model(configs):

    lstm_cell_dimension = configs["lstm_cell_dimension"]
    minibatch_size = configs["minibatch_size"]
    max_epoch_size = configs["max_epoch_size"]
    max_num_of_epochs = configs["max_num_of_epochs"]
    l2_regularization = configs["l2_regularization"]
    gaussian_noise_stdev = configs["gaussian_noise_stdev"]

    print("LSTM Cell Dimension: {}, mbSize: {}, maxEpochSize: {}, maxNumOfEpochs: {}, "
          "l2_regularization: {}, gaussian_noise_std: {}".
          format(lstm_cell_dimension, minibatch_size, max_epoch_size, max_num_of_epochs,
                                                                 l2_regularization, gaussian_noise_stdev))

    tf.reset_default_graph()

    tf.set_random_seed(1)

    # adding noise to the input
    input = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
    noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
    input = input + noise
    target = tf.placeholder(dtype=tf.float32, shape=[None, None, OUTPUT_SIZE])

    # placeholder for the sequence lengths
    sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

    # create the model architecture

    # building the encoder network
    encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units = lstm_cell_dimension, use_peepholes = LSTM_USE_PEEPHOLES)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell, inputs = input, sequence_length = sequence_length, dtype = tf.float32)

    # decoder cell of the decoder network
    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_dimension, use_peepholes=LSTM_USE_PEEPHOLES)

    # the final projection layer to convert the output to the desired dimension
    dense_layer = Dense(units=OUTPUT_SIZE, use_bias=BIAS)

    # building the decoder network for training
    with tf.variable_scope('decode'):
        helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs = target, sequence_length=sequence_length, sampling_probability = 0.0)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell, helper = helper, initial_state = encoder_state, output_layer = dense_layer)

        # perform the decoding
        training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = decoder)

    # building the decoder network for inference
    with tf.variable_scope('decode', reuse = tf.AUTO_REUSE):
        helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs = target, sequence_length=sequence_length, sampling_probability = 1.0)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell, helper = helper,
                                                  initial_state = encoder_state, output_layer = dense_layer)

        # perform the decoding
        inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = decoder)


    # error that should be minimized in the training process
    error = l1_loss(training_decoder_outputs[0], target)

    # l2 regularization of the trainable model parameters
    l2_loss = 0.0
    for var in tf.trainable_variables() :
        l2_loss += tf.nn.l2_loss(var)

    l2_loss = tf.multiply(l2_regularization, l2_loss)

    total_loss = error + l2_loss

    # create the cocob optimizer
    optimizer = cocob_optimizer.COCOB().minimize(loss = total_loss)

    # create the training and validation datasets from the tfrecord files
    training_dataset = tf.data.TFRecordDataset(filenames = [binary_train_file_path], compression_type = "ZLIB")
    validation_dataset = tf.data.TFRecordDataset(filenames = [binary_validation_file_path], compression_type = "ZLIB")

    # parse the records
    training_dataset = training_dataset.map(parser)
    validation_dataset = validation_dataset.map(parser)

    # define the expected shapes of data after padding
    padded_shapes = ([], [tf.Dimension(None), INPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE + 1])

    INFO_FREQ = 1
    smape_final_list = []

    # setup variable initialization
    init_op = tf.global_variables_initializer()

    with tf.Session() as session :
        session.run(init_op)

        for epoch in range(max_num_of_epochs):
            smape_epoch_list = []
            print("Epoch->", epoch)

            # randomly shuffle the time series within the dataset
            training_dataset.shuffle(minibatch_size)

            for epochsize in range(max_epoch_size):
                smape_epochsize__list = []
                padded_training_data_batches = training_dataset.padded_batch(batch_size=minibatch_size, padded_shapes=padded_shapes)

                training_data_batch_iterator = padded_training_data_batches.make_one_shot_iterator()
                next_training_data_batch = training_data_batch_iterator.get_next()

                while True:
                    try:
                        training_data_batch_value = session.run(next_training_data_batch)
                        # sequence_length = session.run(tf.to_int32(next_training_data_batch[0]))
                        session.run(optimizer,
                                    feed_dict={input: training_data_batch_value[1],
                                               target: training_data_batch_value[2],
                                               sequence_length: training_data_batch_value[0].astype(np.int32)})
                    except tf.errors.OutOfRangeError:
                        break

                if epoch % INFO_FREQ == 0:
                    # create a single batch from all the validation time series by padding the datasets to make the variable sequence lengths fixed
                    padded_validation_dataset = validation_dataset.padded_batch(batch_size = minibatch_size, padded_shapes = padded_shapes)

                    # get an iterator to the validation data
                    validation_data_iterator = padded_validation_dataset.make_one_shot_iterator()

                    while True:
                        try:
                            # access the validation data using the iterator
                            next_validation_data_batch = validation_data_iterator.get_next()

                            # get the batch of validation inputs
                            validation_data_batch_value = session.run(next_validation_data_batch)

                            # shape for the target data
                            target_data_shape = [np.shape(validation_data_batch_value[1])[0], np.shape(validation_data_batch_value[1])[1], OUTPUT_SIZE]

                            # get the output of the network for the validation input data batch
                            validation_output = session.run(inference_decoder_outputs[0], feed_dict={input: validation_data_batch_value[1],
                                                                                                     target: np.zeros(target_data_shape),
                                                                                                     sequence_length: validation_data_batch_value[0]
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
                            smape = np.mean(np.abs(converted_validation_output - converted_actual_values) /
                                                (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
                            smape_epochsize__list.append(smape)

                        except tf.errors.OutOfRangeError:
                            break

                smape_epoch_size = np.mean(smape_epochsize__list)
                smape_epoch_list.append(smape_epoch_size)

            smape_epoch = np.mean(smape_epoch_list)
            smape_final_list.append(smape_epoch)

        smape_final = np.mean(smape_final_list)
        print("SMAPE value: {}".format(smape_final))

    return smape_final

if __name__ == '__main__':

    # Build Configuration Space which defines all parameters and their ranges
    configuration_space = ConfigurationSpace()

    lstm_cell_dimension = UniformIntegerHyperparameter("lstm_cell_dimension", 20, 50, default_value = 50)
    minibatch_size = UniformIntegerHyperparameter("minibatch_size", 10, 30, default_value = 10)
    max_epoch_size = UniformIntegerHyperparameter("max_epoch_size", 1, 3, default_value = 1)
    max_num_of_epochs = UniformIntegerHyperparameter("max_num_of_epochs", 3, 20, default_value = 3)
    l2_regularization = UniformFloatHyperparameter("l2_regularization", 0.0001, 0.0008, default_value = 0.0001)
    gaussian_noise_stdev = UniformFloatHyperparameter("gaussian_noise_stdev", 0.0001, 0.0008, default_value = 0.0001)

    configuration_space.add_hyperparameters([lstm_cell_dimension, minibatch_size, max_epoch_size, max_num_of_epochs,
                                             l2_regularization, gaussian_noise_stdev])

    # creating the scenario object
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": 50,
        "cs": configuration_space,
        "deterministic": True,
        "output_dir": "Logs"
    })

    # optimize using an SMAC object
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(0), tae_runner=train_model)

    incumbent = smac.optimize()

    smape_error = train_model(incumbent)

    print("Optimized configuration: {}".format(incumbent))
    print("Optimized Value: {}".format(smape_error))

