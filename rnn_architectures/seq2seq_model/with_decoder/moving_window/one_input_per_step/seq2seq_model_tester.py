import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader as TFRecordReaderMovingWindow
from tfrecords_handler.non_moving_window.tfrecord_reader import TFRecordReader as TFRecordReaderNonMovingWindow
from configs.global_configs import training_data_configs

class Seq2SeqModelTester:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__input_size = kwargs["input_size"]
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
        training_input = tf.placeholder(dtype=tf.float32, shape=[None, self.__input_size, 1])
        testing_input = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        noise = tf.random_normal(shape=tf.shape(training_input), mean=0.0, stddev=gaussian_noise_stdev,
                                 dtype=tf.float32)
        training_input = training_input + noise
        target = tf.placeholder(dtype=tf.float32, shape=[None, self.__output_size, 1])

        # placeholder for the sequence lengths
        input_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])
        output_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        # initial state of the encoder
        encoder_initial_state = tf.placeholder(dtype=tf.float32,
                                               shape=[int(num_hidden_layers), 2, None, int(lstm_cell_dimension)])
        layerwise_encoder_initial_state = tf.unstack(encoder_initial_state, axis=0)
        encoder_initial_state_tuple = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(layerwise_encoder_initial_state[layer][0],
                                           layerwise_encoder_initial_state[layer][1])
             for layer in range(int(num_hidden_layers))]
        )

        # create the model architecture

        # RNN with the LSTM layer
        def lstm_cell():
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(lstm_cell_dimension), use_peepholes=self.__use_peepholes)
            return lstm_cell

        # building the encoder network

        # create two encoders for the training and inference stages
        multi_layered_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            cells=[lstm_cell() for _ in range(int(num_hidden_layers))])

        with tf.variable_scope('encode'):
            _, training_encoder_final_state = tf.nn.dynamic_rnn(cell=multi_layered_encoder_cell,
                                                                initial_state=encoder_initial_state_tuple,
                                                                inputs=training_input,
                                                                sequence_length=input_sequence_length,
                                                                dtype=tf.float32)

        with tf.variable_scope('encode', reuse=tf.AUTO_REUSE):
            _, inference_encoder_final_state = tf.nn.dynamic_rnn(cell=multi_layered_encoder_cell,
                                                                 initial_state=encoder_initial_state_tuple,
                                                                 inputs=testing_input,
                                                                 sequence_length=input_sequence_length,
                                                                 dtype=tf.float32)

        # decoder cell of the decoder network
        multi_layered_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell() for _ in range(int(num_hidden_layers))])

        # the final projection layer to convert the output to the desired dimension
        dense_layer = Dense(units=1, use_bias=self.__use_bias)

        # building the decoder network for training
        with tf.variable_scope('decode'):
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=target, sequence_length=output_sequence_length,
                                                                      sampling_probability=0.0)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=multi_layered_decoder_cell, helper=helper, initial_state=training_encoder_final_state,
                                                      output_layer=dense_layer)

            # perform the decoding
            training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

        # building the decoder network for inference
        with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=target, sequence_length=output_sequence_length,
                                                                      sampling_probability=1.0)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=multi_layered_decoder_cell, helper=helper,
                                                      initial_state=inference_encoder_final_state, output_layer=dense_layer)

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
        tfrecord_reader_with_moving_window = TFRecordReaderMovingWindow(self.__input_size, self.__output_size)
        tfrecord_reader_with_non_moving_window = TFRecordReaderNonMovingWindow()

        # preparing the training data
        # randomly shuffle the time series within the dataset
        training_dataset.shuffle(buffer_size=training_data_configs.SHUFFLE_BUFFER_SIZE)
        training_dataset = training_dataset.map(tfrecord_reader_with_moving_window.validation_data_parser)
        training_dataset.repeat(int(max_epoch_size))

        # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
        padded_training_data_batches = training_dataset.padded_batch(batch_size=int(minibatch_size),
                                                                     padded_shapes=([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size],
                                                                                    [tf.Dimension(None), self.__output_size + 1]))

        # get an iterator to the batches
        training_data_batch_iterator = padded_training_data_batches.make_initializable_iterator()

        # access each batch using the iterator
        next_training_data_batch = training_data_batch_iterator.get_next()

        # preparing the test data
        test_dataset = test_dataset.map(tfrecord_reader_with_non_moving_window.test_data_parser)

        # create a single batch from all the test time series by padding the datasets to make the variable sequence lengths fixed
        padded_test_input_data = test_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes=(
        [], [tf.Dimension(None), 1], [self.__output_size + 1, 1]))

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

                session.run(training_data_batch_iterator.initializer)

                while True:
                    try:
                        training_data_batch_value = session.run(next_training_data_batch)
                        current_minibatch_size = np.shape(training_data_batch_value[1])[0]

                        encoder_initial_state_value = np.zeros(shape=(int(num_hidden_layers), 2, current_minibatch_size, int(lstm_cell_dimension)), dtype=np.float32)

                        # loop through the rolling window batches for each mini-batch
                        for i in range(training_data_batch_value[1].shape[1]):
                            # check for the sequence length of each sequence to check if the sequence has ended
                            length_comparison_array = np.greater([i + 1] * np.shape(training_data_batch_value[1])[0], training_data_batch_value[0])
                            input_sequence_length_values = np.where(length_comparison_array, 0, self.__input_size)
                            output_sequence_length_values = np.where(length_comparison_array, 0, self.__output_size)

                            encoder_initial_state_value, _ = session.run([training_encoder_final_state, optimizer],
                                                                         feed_dict={training_input: np.expand_dims(training_data_batch_value[1][:, i, :], axis=2),
                                                                                  target: np.expand_dims(training_data_batch_value[2][:, i, :], axis=2),
                                                                                  encoder_initial_state: encoder_initial_state_value,
                                                                                  input_sequence_length: input_sequence_length_values,
                                                                                  output_sequence_length: output_sequence_length_values
                                                                                  })
                    except tf.errors.OutOfRangeError:
                        break

            # applying the model to the test data

            list_of_forecasts = []
            while True:
                try:

                    # get the batch of test inputs
                    test_input_batch_value = session.run(test_input_data_batch)

                    # shape for the target data
                    target_data_shape = [np.shape(test_input_batch_value[1])[0], self.__output_size, 1]

                    current_minibatch_size = np.shape(test_input_batch_value[1])[0]

                    encoder_initial_state_value = np.zeros(shape=(int(num_hidden_layers), 2, current_minibatch_size, int(lstm_cell_dimension)), dtype=np.float32)

                    # get the output of the network for the test input data batch
                    test_output = session.run(inference_decoder_outputs[0],
                                              feed_dict={testing_input: test_input_batch_value[1],
                                                         target: np.zeros(shape = target_data_shape),
                                                         encoder_initial_state: encoder_initial_state_value,
                                                         input_sequence_length: test_input_batch_value[0],
                                                         output_sequence_length: [self.__output_size] * np.shape(test_input_batch_value[1])[0]})

                    forecasts = test_output
                    list_of_forecasts.extend(forecasts.tolist())

                except tf.errors.OutOfRangeError:
                    break

            return np.squeeze(list_of_forecasts, axis = 2) #the third dimension is squeezed since it is one