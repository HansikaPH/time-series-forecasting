import numpy as np
import tensorflow as tf
from tfrecords_handler.tfrecord_reader import TFRecordReader

class StackingModelTrainer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__input_size = kwargs["input_size"]
        self.__output_size = kwargs["output_size"]
        self.__binary_train_file_path = kwargs["binary_train_file_path"]
        self.__binary_validation_file_path = kwargs["binary_validation_file_path"]
        self.__contain_zero_values = kwargs["contain_zero_values"]


    def __l1_loss(self, z, t):
        loss = tf.reduce_mean(tf.abs(t - z))
        return loss

    # Training the time series
    def train_model(self, **kwargs):

        # extract the parameters from the kwargs
        num_hidden_layers = kwargs['num_hidden_layers']
        lstm_cell_dimension = kwargs['lstm_cell_dimension']
        minibatch_size = kwargs['minibatch_size']
        max_epoch_size = kwargs['max_epoch_size']
        max_num_epochs = kwargs['max_num_epochs']
        l2_regularization = kwargs['l2_regularization']
        gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        optimizer_fn = kwargs['optimizer_fn']

        tf.reset_default_graph()

        tf.set_random_seed(1)

        # declare the input and output placeholders
        input = tf.placeholder(dtype = tf.float32, shape = [None, None, self.__input_size])
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
        input = input + noise

        true_output = tf.placeholder(dtype = tf.float32, shape = [None, None, self.__output_size])
        sequence_lengths = tf.placeholder(dtype=tf.int64, shape=[None])

        # RNN with the LSTM layer
        def lstm_cell():
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(lstm_cell_dimension), use_peepholes=self.__use_peepholes)
            return lstm_cell

        multi_layered_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell() for _ in range(int(num_hidden_layers))])
        rnn_outputs, states = tf.nn.dynamic_rnn(cell=multi_layered_cell, inputs=input, sequence_length=sequence_lengths,
                                                dtype=tf.float32)

        # connect the dense layer to the RNN
        prediction_output = tf.layers.dense(inputs=tf.convert_to_tensor(value=rnn_outputs, dtype=tf.float32),
                                      units=self.__output_size,
                                      use_bias=self.__use_bias)

        # error that should be minimized in the training process
        error = self.__l1_loss(prediction_output, true_output)

        # l2 regularization of the trainable model parameters
        l2_loss = 0.0
        for var in tf.trainable_variables():
            l2_loss += tf.nn.l2_loss(var)

        l2_loss = tf.multiply(tf.cast(l2_regularization, dtype=tf.float64), tf.cast(l2_loss, dtype=tf.float64))

        total_loss = tf.cast(error, dtype=tf.float64) + l2_loss

        # create the adagrad optimizer
        optimizer = optimizer_fn(total_loss)

        # create the training and validation datasets from the tfrecord files
        training_dataset = tf.data.TFRecordDataset(filenames = [self.__binary_train_file_path], compression_type = "ZLIB")
        validation_dataset = tf.data.TFRecordDataset(filenames = [self.__binary_validation_file_path], compression_type = "ZLIB")

        # parse the records
        tfrecord_reader = TFRecordReader(self.__input_size, self.__output_size)
        training_dataset = training_dataset.map(tfrecord_reader.train_data_parser)
        validation_dataset = validation_dataset.map(tfrecord_reader.validation_data_parser)

        # define the expected shapes of data after padding
        train_padded_shapes = ([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size])
        validation_padded_shapes = ([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size], [tf.Dimension(None), self.__output_size + 1])

        INFO_FREQ = 1
        smape_final_list = []

        # setup variable initialization
        init_op = tf.global_variables_initializer()

        with tf.Session() as session :
            session.run(init_op)

            for epoch in range(int(max_num_epochs)):
                smape_epoch_list = []
                print("Epoch->", epoch)

                # randomly shuffle the time series within the dataset
                training_dataset.shuffle(int(minibatch_size))

                for epochsize in range(int(max_epoch_size)):
                    smape_epochsize__list = []
                    padded_training_data_batches = training_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes=train_padded_shapes)

                    training_data_batch_iterator = padded_training_data_batches.make_one_shot_iterator()
                    next_training_data_batch = training_data_batch_iterator.get_next()


                    while True:
                        try:
                            training_data_batch_value = session.run(next_training_data_batch)
                            session.run(optimizer,
                                        feed_dict={input: training_data_batch_value[1],
                                                   true_output: training_data_batch_value[2],
                                                   sequence_lengths: training_data_batch_value[0]})
                        except tf.errors.OutOfRangeError:
                            break

                    if epoch % INFO_FREQ == 0:
                        # create a single batch from all the validation time series by padding the datasets to make the variable sequence lengths fixed
                        padded_validation_dataset = validation_dataset.padded_batch(batch_size = minibatch_size, padded_shapes = validation_padded_shapes)

                        # get an iterator to the validation data
                        validation_data_iterator = padded_validation_dataset.make_one_shot_iterator()

                        while True:
                            try:
                                # access the validation data using the iterator
                                next_validation_data_batch = validation_data_iterator.get_next()

                                # get the batch of validation inputs
                                validation_data_batch_value = session.run(next_validation_data_batch)

                                # get the output of the network for the validation input data batch
                                validation_output = session.run(prediction_output, feed_dict={input: validation_data_batch_value[1],
                                                                                        sequence_lengths: validation_data_batch_value[0]
                                                                                        })

                                # calculate the smape for the validation data using vectorization

                                # convert the data to remove the preprocessing
                                last_indices = validation_data_batch_value[0] - 1
                                array_first_dimension = np.array(range(0, validation_data_batch_value[0].shape[0]))

                                true_seasonality_values = validation_data_batch_value[3][array_first_dimension, last_indices, 1:]
                                level_values = validation_data_batch_value[3][array_first_dimension, last_indices, 0]

                                last_validation_outputs = validation_output[array_first_dimension, last_indices]
                                converted_validation_output = np.exp(true_seasonality_values + level_values[:, np.newaxis] + last_validation_outputs)

                                actual_values = validation_data_batch_value[2][array_first_dimension, last_indices, :]
                                converted_actual_values = np.exp(true_seasonality_values + level_values[:, np.newaxis] + actual_values)

                                if (self.__contain_zero_values): # to compensate for 0 values in data
                                    converted_validation_output = converted_validation_output - 1
                                    converted_actual_values = converted_actual_values - 1

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
        return smape_final

