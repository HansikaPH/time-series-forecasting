import numpy as np
import tensorflow as tf
from tfrecords_handler.non_moving_window.tfrecord_reader import TFRecordReader as NonMovingWindowTFRecordReader
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader as MovingWindowTFRecordReader
from configs.global_configs import training_data_configs
from configs.global_configs import gpu_configs

class Seq2SeqModelTesterWithDenseLayer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__output_size = kwargs["output_size"]
        self.__input_size = kwargs["input_size"]
        self.__binary_train_file_path = kwargs["binary_train_file_path"]
        self.__binary_test_file_path = kwargs["binary_test_file_path"]
        self.__seed = kwargs["seed"]
        self.__cell_type = kwargs["cell_type"]

    def __l1_loss(self, z, t):
        loss = tf.reduce_mean(tf.abs(t - z))
        return loss

    # Training the time series
    def test_model(self, **kwargs):

        # optimized hyperparameters
        num_hidden_layers = int(kwargs['num_hidden_layers'])
        max_num_epochs = int(kwargs['max_num_epochs'])
        max_epoch_size = int(kwargs['max_epoch_size'])
        cell_dimension = int(kwargs['cell_dimension'])
        l2_regularization = kwargs['l2_regularization']
        minibatch_size = int(kwargs['minibatch_size'])
        gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']
        optimizer_fn = kwargs['optimizer_fn']

        # reset the tensorflow graph
        tf.reset_default_graph()

        tf.set_random_seed(self.__seed)

        # declare the input and output placeholders

        # adding noise to the input
        input = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        testing_input = input
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
        training_input = input + noise

        training_outputs = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])
        training_targets = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])

        # placeholder for the sequence lengths
        sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        weight_initializer = tf.truncated_normal_initializer(stddev=random_normal_initializer_stdev)

        # create the model architecture

        # RNN with the layer of cells
        def cell():
            if self.__cell_type == "LSTM":
                cell = tf.nn.rnn_cell.LSTMCell(num_units=int(cell_dimension), use_peepholes=self.__use_peepholes,
                                         initializer=weight_initializer)
            elif self.__cell_type == "GRU":
                cell = tf.nn.rnn_cell.GRUCell(num_units=int(cell_dimension), kernel_initializer=weight_initializer)
            elif self.__cell_type == "RNN":
                cell = tf.nn.rnn_cell.BasicRNNCell(num_units=int(cell_dimension))
            return cell

        # building the encoder network
        multi_layered_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            cells=[cell() for _ in range(int(num_hidden_layers))])

        # define actual_batch_size
        actual_batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        # define the placeholder for the
        training_encoder_initial_state = multi_layered_encoder_cell.zero_state(batch_size=actual_batch_size,
                                                                               dtype=tf.float32)

        with tf.variable_scope('train_encoder_scope') as encoder_train_scope:
            training_encoder_outputs, training_encoder_state = tf.nn.dynamic_rnn(cell=multi_layered_encoder_cell,
                                                                                 inputs=training_input,
                                                                                 sequence_length=sequence_length,
                                                                                 dtype=tf.float32)

        with tf.variable_scope(encoder_train_scope, reuse=tf.AUTO_REUSE) as encoder_inference_scope:
            inference_encoder_outputs, inference_encoder_states = tf.nn.dynamic_rnn(cell=multi_layered_encoder_cell,
                                                                                    inputs=testing_input,
                                                                                    sequence_length=sequence_length,
                                                                                    dtype=tf.float32)

        # create a tensor array for the indices of the encoder outputs array
        new_first_index_array = tf.range(start=0, limit=tf.shape(sequence_length)[0], delta=1)
        new_train_second_index_array = tf.tile([self.__input_size - 1], [tf.shape(sequence_length)[0]])
        train_output_array_indices = tf.stack([new_first_index_array, new_train_second_index_array], axis=-1)
        inference_output_array_indices = tf.stack([new_first_index_array, sequence_length - 1], axis=-1)

        # building the decoder network for training
        with tf.variable_scope('dense_layer_train_scope') as dense_layer_train_scope:
            train_final_timestep_predictions = tf.gather_nd(params=training_encoder_outputs,
                                                            indices=train_output_array_indices)
            # the final projection layer to convert the encoder_outputs to the desired dimension
            train_prediction_output = tf.layers.dense(
                inputs=tf.convert_to_tensor(value=train_final_timestep_predictions, dtype=tf.float32),
                units=self.__output_size,
                use_bias=self.__use_bias, kernel_initializer=weight_initializer)

        # building the decoder network for inference
        with tf.variable_scope(dense_layer_train_scope, reuse=tf.AUTO_REUSE) as dense_layer_inference_scope:
            inference_final_timestep_predictions = tf.gather_nd(params=inference_encoder_outputs,
                                                               indices=inference_output_array_indices)

            # the final projection layer to convert the encoder_outputs to the desired dimension
            inference_prediction_output = tf.layers.dense(
                inputs=tf.convert_to_tensor(value=inference_final_timestep_predictions, dtype=tf.float32),
                units=self.__output_size,
                use_bias=self.__use_bias, kernel_initializer=weight_initializer)


        # error that should be minimized in the training process
        error = self.__l1_loss(training_outputs, training_targets)

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
        non_moving_window_tfrecord_reader = NonMovingWindowTFRecordReader()
        moving_window_tfrecord_reader = MovingWindowTFRecordReader(self.__input_size, self.__output_size)

        # preparing the training data
        # randomly shuffle the time series within the dataset
        shuffle_seed = tf.placeholder(dtype=tf.int64, shape=[])
        training_dataset = training_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=training_data_configs.SHUFFLE_BUFFER_SIZE,
                                                                  count=int(max_epoch_size), seed=shuffle_seed))
        training_dataset = training_dataset.map(moving_window_tfrecord_reader.validation_data_parser)

        # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
        padded_training_data_batches = training_dataset.padded_batch(batch_size=int(minibatch_size),
                                                                     padded_shapes=([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size],
                                                                                    [tf.Dimension(None), self.__output_size + 1]))

        # get an iterator to the batches
        training_data_batch_iterator = padded_training_data_batches.make_initializable_iterator()

        # access each batch using the iterator
        next_training_data_batch = training_data_batch_iterator.get_next()

        # preparing the test data
        test_dataset = test_dataset.map(non_moving_window_tfrecord_reader.test_data_parser)

        # create a single batch from all the test time series by padding the datasets to make the variable sequence lengths fixed
        padded_test_input_data = test_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes=(
        [], [tf.Dimension(None), 1], [self.__output_size + 1, 1]))

        # get an iterator to the test input data batch
        test_input_iterator = padded_test_input_data.make_one_shot_iterator()

        # access the test input batch using the iterator
        test_input_data_batch = test_input_iterator.get_next()

        # setup variable initialization
        init_op = tf.global_variables_initializer()

        # define the GPU options
        gpu_options = tf.GPUOptions(visible_device_list=gpu_configs.visible_device_list, allow_growth=True)

        with tf.Session(
                config=tf.ConfigProto(log_device_placement=gpu_configs.log_device_placement, allow_soft_placement=True,
                                      gpu_options=gpu_options)) as session:
            session.run(init_op)

            # graph plotter object
            # graph_plotter = GraphPlotter(session, 1)

            for epoch in range(int(max_num_epochs)):
                print("Epoch->", epoch)

                session.run(training_data_batch_iterator.initializer, feed_dict={shuffle_seed:epoch})
                while True:
                    try:
                        next_training_batch_value = session.run(next_training_data_batch,
                                                                feed_dict={shuffle_seed: epoch})
                        actual_batch_size_value = np.shape(next_training_batch_value[0])[0]
                        training_encoder_state_value = session.run(training_encoder_initial_state, feed_dict={
                            actual_batch_size: actual_batch_size_value})
                        training_predictions = []

                        for i in range(np.shape(next_training_batch_value[1])[1]):
                            # splitting the input and output batch
                            splitted_input = np.expand_dims(next_training_batch_value[1][:, i, :], axis=2)

                            # get the values of the input and output sequence lengths
                            length_comparison_array = np.greater([i + 1] * np.shape(splitted_input)[0],
                                                                 next_training_batch_value[0])
                            sequence_length_values = np.where(length_comparison_array, 0, self.__input_size)

                            # get the predictions
                            training_prediction_output_values, training_encoder_state_value = session.run(
                                [train_prediction_output, training_encoder_state], feed_dict={
                                    input: splitted_input,
                                    sequence_length: sequence_length_values,
                                    training_encoder_initial_state: training_encoder_state_value
                                })
                            training_prediction_output_values = np.ma.masked_array(training_prediction_output_values)
                            training_prediction_output_values[sequence_length_values == 0] = 0
                            training_predictions.append(training_prediction_output_values)

                        training_predictions = np.transpose(np.reshape(np.array(training_predictions), newshape=(np.shape(next_training_batch_value[1])[1], actual_batch_size_value, self.__output_size)), (1, 0, 2))

                        # backpropagate the accumulated errors
                        session.run(optimizer, feed_dict={
                            training_outputs: np.array(training_predictions),
                            training_targets: next_training_batch_value[2]
                        })
                    except tf.errors.OutOfRangeError:
                        break
                # graph_plotter.plot_train(losses, epoch)
            # applying the model to the test data

            list_of_forecasts = []
            while True:
                try:

                    # get the batch of test inputs
                    test_input_batch_value = session.run(test_input_data_batch)

                    # get the output of the network for the test input data batch
                    test_output = session.run(inference_prediction_output,
                                              feed_dict={input: test_input_batch_value[1],
                                                         sequence_length: test_input_batch_value[0],
                                                         })

                    forecasts = test_output
                    list_of_forecasts.extend(forecasts.tolist())

                except tf.errors.OutOfRangeError:
                    break

            return np.squeeze(list_of_forecasts, axis = 2) #the third dimension is squeezed since it is one