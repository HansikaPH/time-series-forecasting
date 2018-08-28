import numpy as np
import tensorflow as tf
from tfrecords_handler.non_moving_window.tfrecord_reader import TFRecordReader

class Seq2SeqModelTrainerWithDenseLayer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__output_size = kwargs["output_size"]
        self.__binary_train_file_path = kwargs["binary_train_file_path"]
        self.__binary_validation_file_path = kwargs["binary_validation_file_path"]
        self.__contain_zero_values = kwargs["contain_zero_values"]

    def __l1_loss(self, z, t):
        loss = tf.reduce_mean(tf.abs(t - z))
        return loss

    # Training the time series
    def train_model(self, **kwargs):

        num_hidden_layers = kwargs['num_hidden_layers']
        lstm_cell_dimension = kwargs["lstm_cell_dimension"]
        minibatch_size = kwargs["minibatch_size"]
        max_epoch_size = kwargs["max_epoch_size"]
        max_num_epochs = kwargs["max_num_epochs"]
        l2_regularization = kwargs["l2_regularization"]
        gaussian_noise_stdev = kwargs["gaussian_noise_stdev"]
        optimizer_fn = kwargs["optimizer_fn"]

        tf.reset_default_graph()

        tf.set_random_seed(1)

        # adding noise to the input
        input = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
        input = input + noise
        target = tf.placeholder(dtype=tf.float32, shape=[None, self.__output_size, 1])

        # placeholder for the sequence lengths
        input_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        # create the model architecture

        # building the encoder network

        # RNN with the LSTM layer
        def lstm_cell():
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(lstm_cell_dimension), use_peepholes=self.__use_peepholes)
            return lstm_cell

        multi_layered_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell() for _ in range(int(num_hidden_layers))])
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = multi_layered_encoder_cell, inputs = input, sequence_length = input_sequence_length, dtype = tf.float32)

        # create a tensor array for the indices of the encoder outputs array
        new_index_array = tf.range(start=0, limit=tf.shape(input_sequence_length)[0], delta=1)
        output_array_indices = tf.stack([new_index_array, input_sequence_length - 1], axis=-1)
        final_timestep_predictions = tf.gather_nd(params=encoder_outputs, indices=output_array_indices)

        # the final projection layer to convert the encoder_outputs to the desired dimension
        prediction_output = tf.layers.dense(inputs=tf.convert_to_tensor(value=final_timestep_predictions, dtype=tf.float32), units=self.__output_size, use_bias=self.__use_bias)
        prediction_output = tf.expand_dims(input=prediction_output, axis=2)

        # error that should be minimized in the training process
        error = self.__l1_loss(prediction_output, target)

        # l2 regularization of the trainable model parameters
        l2_loss = 0.0
        for var in tf.trainable_variables():
            l2_loss += tf.nn.l2_loss(var)

        l2_loss = tf.multiply(tf.cast(l2_regularization, dtype=tf.float64), tf.cast(l2_loss, dtype=tf.float64))

        total_loss = tf.cast(error, dtype=tf.float64) + l2_loss

        # create the optimizer
        optimizer = optimizer_fn(total_loss)

        # create the training and validation datasets from the tfrecord files
        training_dataset = tf.data.TFRecordDataset(filenames = [self.__binary_train_file_path], compression_type = "ZLIB")
        validation_dataset = tf.data.TFRecordDataset(filenames = [self.__binary_validation_file_path], compression_type = "ZLIB")

        # parse the records
        tfrecord_reader = TFRecordReader()
        training_dataset = training_dataset.map(tfrecord_reader.train_data_parser)
        validation_dataset = validation_dataset.map(tfrecord_reader.validation_data_parser)

        # define the expected shapes of data after padding
        train_padded_shapes = ([], [tf.Dimension(None), 1], [self.__output_size, 1])
        validation_padded_shapes = ([], [tf.Dimension(None), 1], [self.__output_size, 1], [self.__output_size + 1, 1])

        INFO_FREQ = 1
        smape_final_list = []

        # setup variable initialization
        init_op = tf.global_variables_initializer()

        with tf.Session() as session :
            session.run(init_op)

            for epoch in range(max_num_epochs):
                smape_epoch_list = []
                print("Epoch->", epoch)

                # randomly shuffle the time series within the dataset
                training_dataset.shuffle(minibatch_size)

                for epochsize in range(max_epoch_size):
                    padded_training_data_batches = training_dataset.padded_batch(batch_size=minibatch_size, padded_shapes=train_padded_shapes)

                    training_data_batch_iterator = padded_training_data_batches.make_one_shot_iterator()
                    next_training_data_batch = training_data_batch_iterator.get_next()

                    while True:
                        try:
                            training_data_batch_value = session.run(next_training_data_batch)
                            session.run(optimizer,
                                        feed_dict={input: training_data_batch_value[1],
                                                   target: training_data_batch_value[2],
                                                   input_sequence_length: training_data_batch_value[0]
                                                   })
                        except tf.errors.OutOfRangeError:
                            break

                if epoch % INFO_FREQ == 0:
                    # create a single batch from all the validation time series by padding the datasets to make the variable sequence lengths fixed
                    padded_validation_dataset = validation_dataset.padded_batch(batch_size=minibatch_size,
                                                                                padded_shapes=validation_padded_shapes)

                    # get an iterator to the validation data
                    validation_data_iterator = padded_validation_dataset.make_one_shot_iterator()
                    # access the validation data using the iterator
                    next_validation_data_batch = validation_data_iterator.get_next()

                    while True:
                        try:
                            # get the batch of validation inputs
                            validation_data_batch_value = session.run(next_validation_data_batch)

                            # shape for the target data
                            target_data_shape = [np.shape(validation_data_batch_value[1])[0], self.__output_size, 1]

                            # get the output of the network for the validation input data batch
                            validation_output = session.run(prediction_output,
                                feed_dict={input: validation_data_batch_value[1],
                                           target: np.zeros(target_data_shape),
                                           input_sequence_length: validation_data_batch_value[0]
                                           })

                            # calculate the smape for the validation data using vectorization

                            # convert the data to remove the preprocessing
                            true_seasonality_values = validation_data_batch_value[3][:, 1:, 0]
                            level_values = validation_data_batch_value[3][:, 0, 0]

                            converted_validation_output = np.exp(
                                true_seasonality_values + level_values[:, np.newaxis] + np.squeeze(validation_output,
                                                                                                   axis=2))

                            actual_values = validation_data_batch_value[2]
                            converted_actual_values = np.exp(
                                true_seasonality_values + level_values[:, np.newaxis] + np.squeeze(actual_values,
                                                                                                   axis=2))

                            if (self.__contain_zero_values):  # to compensate for 0 values in data
                                converted_validation_output = converted_validation_output - 1
                                converted_actual_values = converted_actual_values - 1

                            # calculate the smape
                            smape = np.mean(np.abs(converted_validation_output - converted_actual_values) /
                                            (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
                            smape_epoch_list.append(smape)

                        except tf.errors.OutOfRangeError:
                            break

                smape_epoch = np.mean(smape_epoch_list)
                smape_final_list.append(smape_epoch)

            smape_final = np.mean(smape_final_list)
            print("SMAPE value: {}".format(smape_final))

        return smape_final