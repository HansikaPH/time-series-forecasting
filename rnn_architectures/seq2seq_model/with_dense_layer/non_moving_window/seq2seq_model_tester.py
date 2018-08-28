import numpy as np
import tensorflow as tf
from tfrecords_handler.non_moving_window.tfrecord_reader import TFRecordReader

class Seq2SeqModelTesterWithDenseLayer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__output_size = kwargs["output_size"]
        self.__binary_train_file_path = kwargs["binary_train_file_path"]
        self.__binary_test_file_path = kwargs["binary_test_file_path"]

    def __l1_loss(self, z, t):
        loss = tf.reduce_mean(tf.abs(t - z))
        return loss

    # Training the time series
    def test_model(self, **kwargs):

        # optimized hyperparameters
        num_hidden_layers = kwargs['num_hidden_layers']
        max_num_epochs = kwargs['max_num_epochs']
        max_epoch_size = kwargs['max_epoch_size']
        lstm_cell_dimension = kwargs['lstm_cell_dimension']
        l2_regularization = kwargs['l2_regularization']
        minibatch_size = kwargs['minibatch_size']
        gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        optimizer_fn = kwargs['optimizer_fn']

        # reset the tensorflow graph
        tf.reset_default_graph()

        tf.set_random_seed(1)

        # declare the input and output placeholders

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
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=multi_layered_encoder_cell, inputs=input, sequence_length=input_sequence_length,
                                                           dtype=tf.float32)

        # create a tensor array for the indices of the encoder outputs array
        new_index_array = tf.range(start=0, limit=tf.shape(input_sequence_length)[0], delta=1)
        output_array_indices = tf.stack([new_index_array, input_sequence_length - 1], axis=-1)
        final_timestep_predictions = tf.gather_nd(params=encoder_outputs, indices=output_array_indices)

        # the final projection layer to convert the encoder_outputs to the desired dimension
        prediction_output = tf.layers.dense(
            inputs=tf.convert_to_tensor(value=final_timestep_predictions, dtype=tf.float32), units=self.__output_size,
            use_bias=self.__use_bias)
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

        # create the Dataset objects for the training and test data
        training_dataset = tf.data.TFRecordDataset(filenames = [self.__binary_train_file_path], compression_type = "ZLIB")
        test_dataset = tf.data.TFRecordDataset([self.__binary_test_file_path], compression_type = "ZLIB")

        # parse the records
        tfrecord_reader = TFRecordReader()
        training_dataset = training_dataset.map(tfrecord_reader.train_data_parser)
        test_dataset = test_dataset.map(tfrecord_reader.test_data_parser)

        # setup variable initialization
        init_op = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init_op)

            for epoch in range(int(max_num_epochs)):
                print("Epoch->", epoch)

                # randomly shuffle the time series within the dataset
                training_dataset.shuffle(int(minibatch_size))

                for epochsize in range(int(max_epoch_size)):

                    # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
                    padded_training_data_batches = training_dataset.padded_batch(batch_size = int(minibatch_size),
                                          padded_shapes = ([], [tf.Dimension(None), 1], [self.__output_size, 1]))

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
                                                   input_sequence_length: next_training_batch_value[0],
                                                   })
                        except tf.errors.OutOfRangeError:
                            break

            # applying the model to the test data

            # create a single batch from all the test time series by padding the datasets to make the variable sequence lengths fixed
            padded_test_input_data = test_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes = ([], [tf.Dimension(None), 1], [self.__output_size + 1, 1]))

            # get an iterator to the test input data batch
            test_input_iterator = padded_test_input_data.make_one_shot_iterator()

            # access the test input batch using the iterator
            test_input_data_batch = test_input_iterator.get_next()

            list_of_forecasts = []
            while True:
                try:

                    # get the batch of test inputs
                    test_input_batch_value = session.run(test_input_data_batch)

                    # shape for the target data
                    target_data_shape = [np.shape(test_input_batch_value[1])[0], self.__output_size, 1]

                    # get the output of the network for the test input data batch
                    test_output = session.run(prediction_output,
                                              feed_dict={input: test_input_batch_value[1],
                                                         target: np.zeros(shape = target_data_shape),
                                                         input_sequence_length: test_input_batch_value[0],
                                                         })

                    forecasts = test_output
                    list_of_forecasts.extend(forecasts.tolist())

                except tf.errors.OutOfRangeError:
                    break

            return np.squeeze(list_of_forecasts, axis = 2) #the third dimension is squeezed since it is one