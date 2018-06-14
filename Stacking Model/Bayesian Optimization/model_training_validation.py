import numpy as np
from bayes_opt import BayesianOptimization

# import the config space and the different types of parameters
from smac.facade.func_facade import fmin_smac

import tensorflow as tf

# Input/Output Window sizes
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Training and Validation file paths.
binary_train_file_path = '../../DataSets/CIF 2016/Binary Files/stl_12i15.tfrecords'
binary_validation_file_path = '../../DataSets/CIF 2016/Binary Files/stl_12i15v.tfrecords'

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

def create_encoder_decoder_model(lstm_cell_dimension, gaussian_noise_stdev):
    # declare the input and output placeholders
    input = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
    noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
    input = input + noise

    label = tf.placeholder(dtype=tf.float32, shape=[None, None, OUTPUT_SIZE])
    sequence_lengths = tf.placeholder(dtype=tf.int64, shape=[None])

    # create the encoder network
    encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units = int(lstm_cell_dimension), use_peepholes = LSTM_USE_PEEPHOLES)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=input, sequence_length=sequence_lengths,
                                            dtype=tf.float32)

    # create the decoder network



# def create_stacking_model():


# Training the time series
def train_model(learning_rate, lstm_cell_dimension, minibatch_size, max_epoch_size, max_num_of_epochs, l2_regularization, gaussian_noise_stdev):

    print("Learning Rate: {}, LSTM Cell Dimension: {}, mbSize: {}, maxEpochSize: {}, maxNumOfEpochs: {}, "
          "l2_regularization: {}, gaussian_noise_std: {}".format(learning_rate, lstm_cell_dimension, minibatch_size, max_epoch_size, max_num_of_epochs, l2_regularization, gaussian_noise_stdev))

    tf.reset_default_graph()

    tf.set_random_seed(1)

    # declare the input and output placeholders
    input = tf.placeholder(dtype = tf.float32, shape = [None, None, INPUT_SIZE])
    noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
    input = input + noise

    label = tf.placeholder(dtype = tf.float32, shape = [None, None, OUTPUT_SIZE])
    sequence_lengths = tf.placeholder(dtype=tf.int64, shape=[None])

    # create the model architecture

    # RNN with the LSTM layer
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = int(lstm_cell_dimension), use_peepholes = LSTM_USE_PEEPHOLES)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = input, sequence_length = sequence_lengths, dtype = tf.float32)

    # connect the dense layer to the RNN
    dense_layer = tf.layers.dense(inputs = tf.convert_to_tensor(value = rnn_outputs, dtype = tf.float32), units = OUTPUT_SIZE, use_bias = BIAS)

    # error that should be minimized in the training process
    error = l1_loss(dense_layer, label)

    # l2 regularization of the trainable model parameters
    l2_loss = 0.0
    for var in tf.trainable_variables() :
        l2_loss += tf.nn.l2_loss(var)

    l2_loss = tf.multiply(l2_regularization, tf.cast(l2_loss, tf.float64))

    total_loss = tf.cast(error, tf.float64) + l2_loss

    # create the adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(total_loss)

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

        for epoch in range(int(max_num_of_epochs)):
            smape_epoch_list = []
            print("Epoch->", epoch)

            # randomly shuffle the time series within the dataset
            training_dataset.shuffle(int(minibatch_size))

            for epochsize in range(int(max_epoch_size)):
                smape_epochsize__list = []
                padded_training_data_batches = training_dataset.padded_batch(batch_size=10, padded_shapes=padded_shapes)

                training_data_batch_iterator = padded_training_data_batches.make_one_shot_iterator()
                next_training_data_batch = training_data_batch_iterator.get_next()


                while True:
                    try:
                        training_data_batch_value = session.run(next_training_data_batch)
                        total_loss_value, _ = session.run([total_loss, optimizer],
                                    feed_dict={input: training_data_batch_value[1],
                                               label: training_data_batch_value[2],
                                               sequence_lengths: training_data_batch_value[0]})
                        print("Total Loss: ", total_loss_value)
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
        max_value = 1 / (smape_final)

    return max_value

if __name__ == '__main__':

    init_points = 2
    num_iter = 30

    # using bayesian optimizer for hyperparameter optimization
    bayesian_optimization = BayesianOptimization(train_model, {'learning_rate': (0.0001, 0.0008),
                                                                'lstm_cell_dimension': (50, 100),
                                                                'minibatch_size': (10, 30),
                                                                'max_epoch_size': (1, 3),
                                                                'max_num_of_epochs': (3, 20),
                                                                'l2_regularization': (0.0001, 0.0008),
                                                                'gaussian_noise_stdev': (0.0001, 0.0008)
                                                            })

    bayesian_optimization.maximize(init_points = init_points, n_iter = num_iter)

