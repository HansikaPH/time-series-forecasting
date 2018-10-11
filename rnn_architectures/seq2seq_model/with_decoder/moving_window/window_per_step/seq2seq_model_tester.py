import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader
from configs.global_configs import training_data_configs

class Seq2SeqModelTester:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__input_size = kwargs["input_size"]
        self.__output_size = kwargs["output_size"]
        self.__binary_train_file_path = kwargs["binary_train_file_path"]
        self.__binary_test_file_path = kwargs["binary_test_file_path"]
        self.__seed = kwargs["seed"]

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
        random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']
        optimizer_fn = kwargs['optimizer_fn']

        # reset the tensorflow graph
        tf.reset_default_graph()

        tf.set_random_seed(self.__seed)

        # declare the input and output placeholders

        # adding noise to the input
        input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__input_size])
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
        input = input + noise
        target = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])

        # placeholder for the sequence lengths
        sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        weight_initializer = tf.truncated_normal_initializer(stddev=random_normal_initializer_stdev, seed=self.__seed)

        # create the model architecture

        # building the encoder network

        # RNN with the LSTM layer
        def lstm_cell():
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(lstm_cell_dimension), use_peepholes=self.__use_peepholes, initializer=weight_initializer)
            return lstm_cell

        multi_layered_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell() for _ in range(int(num_hidden_layers))])
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=multi_layered_encoder_cell, inputs=input, sequence_length=sequence_length,
                                                           dtype=tf.float32)

        # decoder cell of the decoder network
        multi_layered_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell() for _ in range(int(num_hidden_layers))])

        # the final projection layer to convert the output to the desired dimension
        dense_layer = Dense(units=self.__output_size, use_bias=self.__use_bias, kernel_initializer=weight_initializer)

        # building the decoder network for training
        with tf.variable_scope('decode'):
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=target, sequence_length=sequence_length,
                                                                      sampling_probability=0.0)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=multi_layered_decoder_cell, helper=helper, initial_state=encoder_state,
                                                      output_layer=dense_layer)

            # perform the decoding
            training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

        # building the decoder network for inference
        with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=target, sequence_length=sequence_length,
                                                                      sampling_probability=1.0)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=multi_layered_decoder_cell, helper=helper,
                                                      initial_state=encoder_state, output_layer=dense_layer)

            # perform the decoding
            inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

        # error that should be minimized in the training process
        error = self.__l1_loss(training_decoder_outputs[0], target)

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
        tfrecord_reader = TFRecordReader(input_size=self.__input_size, output_size=self.__output_size)

        # preparing the training data
        # randomly shuffle the time series within the dataset
        shuffle_seed = tf.placeholder(dtype=tf.int64, shape=[])
        training_dataset = training_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=training_data_configs.SHUFFLE_BUFFER_SIZE,
                                                                  count=int(max_epoch_size), seed=shuffle_seed))
        training_dataset = training_dataset.map(tfrecord_reader.validation_data_parser)

        # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
        padded_training_data_batches = training_dataset.padded_batch(batch_size=int(minibatch_size),
                                                                     padded_shapes=(
                                                                         [], [tf.Dimension(None), self.__input_size],
                                                                         [tf.Dimension(None), self.__output_size],
                                                                         [tf.Dimension(None), self.__output_size + 1]))

        # get an iterator to the batches
        training_data_batch_iterator = padded_training_data_batches.make_initializable_iterator()

        # access each batch using the iterator
        next_training_data_batch = training_data_batch_iterator.get_next()

        # preparing the test data
        test_dataset = test_dataset.map(tfrecord_reader.test_data_parser)

        # create a single batch from all the test time series by padding the datasets to make the variable sequence lengths fixed
        padded_test_input_data = test_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes=([], [tf.Dimension(None), self.__input_size],
                                                                                                          [tf.Dimension(None), self.__output_size + 1]))

        # get an iterator to the test input data batch
        test_input_iterator = padded_test_input_data.make_one_shot_iterator()

        # access the test input batch using the iterator
        test_input_data_batch = test_input_iterator.get_next()

        # setup variable initialization
        init_op = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init_op)

            for epoch in range(int(max_num_epochs)):
                print("Epoch->", epoch)

                session.run(training_data_batch_iterator.initializer, feed_dict={shuffle_seed:epoch})

                while True:
                    try:
                        next_training_batch_value = session.run(next_training_data_batch, feed_dict={shuffle_seed:epoch})

                        # model training
                        session.run(optimizer,
                                    feed_dict={input: next_training_batch_value[1],
                                               target: next_training_batch_value[2],
                                               sequence_length: next_training_batch_value[0]})
                    except tf.errors.OutOfRangeError:
                        break

            # applying the model to the test data


            list_of_forecasts = []
            while True:
                try:

                    # get the batch of test inputs
                    test_input_batch_value = session.run(test_input_data_batch)

                    # shape for the target data
                    target_data_shape = [np.shape(test_input_batch_value[1])[0], np.shape(test_input_batch_value[1])[1], self.__output_size]

                    # get the output of the network for the test input data batch
                    test_output = session.run(inference_decoder_outputs[0],
                                              feed_dict={input: test_input_batch_value[1],
                                                         target: np.zeros(shape = target_data_shape),
                                                         sequence_length: test_input_batch_value[0]})

                    forecasts = test_output
                    list_of_forecasts.extend(forecasts.tolist())

                except tf.errors.OutOfRangeError:
                    break

            return list_of_forecasts