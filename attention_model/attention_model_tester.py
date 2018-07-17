import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tfrecords_handler.tfrecord_reader import TFRecordReader

class AttentionModelTester:

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
        input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__input_size])
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
        input = input + noise
        target = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])

        # placeholder for the sequence lengths
        sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        # create the model architecture

        # building the encoder network
        encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(lstm_cell_dimension), use_peepholes=self.__use_peepholes)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=input, sequence_length=sequence_length,
                                                           dtype=tf.float32)

        # creating an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=lstm_cell_dimension, memory=encoder_outputs,
                                                                   memory_sequence_length=sequence_length)

        # decoder cell of the decoder network
        decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_dimension, use_peepholes=self.__use_peepholes)

        # using the attention wrapper to wrap the decoding cell
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                           attention_layer_size=lstm_cell_dimension)
        # the final projection layer to convert the output to the desired dimension
        dense_layer = Dense(units=self.__output_size, use_bias=self.__use_bias)

        # create the initial state for the decoder
        decoder_initial_state = decoder_cell.zero_state(batch_size=tf.shape(input)[0], dtype=tf.float32).clone(
            cell_state=encoder_state)

        # building the decoder network for training
        with tf.variable_scope('decode'):
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=target, sequence_length=sequence_length,
                                                                      sampling_probability=0.0)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=decoder_initial_state,
                                                      output_layer=dense_layer)

            # perform the decoding
            training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

        # building the decoder network for inference
        with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=target, sequence_length=sequence_length,
                                                                      sampling_probability=1.0)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                      initial_state=decoder_initial_state, output_layer=dense_layer)

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
        tfrecord_reader = TFRecordReader(self.__input_size, self.__output_size)
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
                                          padded_shapes = ([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size]))

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
            padded_test_input_data = test_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes = ([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size + 1]))

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
                    target_data_shape = [np.shape(test_input_batch_value[1])[0], np.shape(test_input_batch_value[1])[1], self.__output_size]

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