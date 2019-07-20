import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader as MovingWindowTFRecordReader
from tfrecords_handler.non_moving_window.tfrecord_reader import TFRecordReader as NonMovingWindowTFRecordReader
from configs.global_configs import model_training_configs
from configs.global_configs import training_data_configs
from configs.global_configs import gpu_configs

class AttentionModelTrainer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__output_size = kwargs["output_size"]
        self.__input_size = kwargs["input_size"]
        self.__binary_train_file_path = kwargs["binary_train_file_path"]
        self.__binary_validation_file_path = kwargs["binary_validation_file_path"]
        self.__contain_zero_values = kwargs["contain_zero_values"]
        self.__address_near_zero_instability = kwargs["address_near_zero_instability"]
        self.__integer_conversion = kwargs["integer_conversion"]
        self.__seed = kwargs["seed"]
        self.__cell_type = kwargs["cell_type"]

    def __l1_loss(self, z, t):
        loss = tf.reduce_mean(tf.abs(t - z))
        return loss

    # Training the time series
    def train_model(self, **kwargs):

        num_hidden_layers = int(kwargs['num_hidden_layers'])
        cell_dimension = int(kwargs["cell_dimension"])
        minibatch_size = int(kwargs["minibatch_size"])
        max_epoch_size = int(kwargs["max_epoch_size"])
        max_num_epochs = int(kwargs["max_num_epochs"])
        l2_regularization = kwargs["l2_regularization"]
        gaussian_noise_stdev = kwargs["gaussian_noise_stdev"]
        random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']
        optimizer_fn = kwargs["optimizer_fn"]

        tf.reset_default_graph()

        tf.set_random_seed(self.__seed)

        # adding noise to the input
        input = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
        training_input = input + noise

        validation_input = input

        training_outputs = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])
        training_targets = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])

        decoder_input = tf.placeholder(dtype=tf.float32, shape=[None, self.__output_size, 1])

        # placeholder for the sequence lengths
        input_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])
        output_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

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

        training_encoder_initial_state = multi_layered_encoder_cell.zero_state(batch_size=actual_batch_size,
                                                                               dtype=tf.float32)

        with tf.variable_scope('train_encoder_scope') as encoder_train_scope:
            training_encoder_outputs, training_encoder_state = tf.nn.dynamic_rnn(cell=multi_layered_encoder_cell,
                                                                                 inputs=training_input,
                                                                                 initial_state=training_encoder_initial_state,
                                                                                 sequence_length=input_sequence_length,
                                                                                 dtype=tf.float32)

        with tf.variable_scope(encoder_train_scope, reuse=tf.AUTO_REUSE) as encoder_inference_scope:
            inference_encoder_outputs, inference_encoder_state = tf.nn.dynamic_rnn(cell=multi_layered_encoder_cell,
                                                                                    inputs=validation_input,
                                                                                    sequence_length=input_sequence_length,
                                                                                    dtype=tf.float32)

        # the final projection layer to convert the output to the desired dimension
        dense_layer = Dense(units=1, use_bias=self.__use_bias, kernel_initializer=weight_initializer)

        # decoder cell of the decoder network
        multi_layered_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            cells=[cell() for _ in range(int(num_hidden_layers))])

        # building the decoder network for training
        with tf.variable_scope('decoder_train_scope') as decoder_train_scope:
            # creating an attention layer
            training_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=cell_dimension,
                                                                                memory=training_encoder_outputs)
            # using the attention wrapper to wrap the decoding cell
            training_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=multi_layered_decoder_cell,
                                                                        attention_mechanism=training_attention_mechanism,
                                                                        attention_layer_size=cell_dimension)
            # create the initial state for the decoder
            training_decoder_initial_state = training_decoder_cell.zero_state(batch_size=tf.shape(input)[0],
                                                                              dtype=tf.float32).clone(
                cell_state=training_encoder_state)
            training_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=decoder_input,
                                                                               sequence_length=output_sequence_length,
                                                                               sampling_probability=0.0)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=training_decoder_cell, helper=training_helper,
                                                               initial_state=training_decoder_initial_state,
                                                               output_layer=dense_layer)

            # perform the decoding
            training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder)

        inference_attention_memory = tf.placeholder(dtype=tf.float32, shape=[None, self.__input_size, cell_dimension])

        # building the decoder network for inference
        with tf.variable_scope(decoder_train_scope, reuse=tf.AUTO_REUSE) as decoder_inference_scope:
            # creating an attention layer
            inference_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=cell_dimension,
                                                                                 memory=inference_attention_memory)
            inference_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=multi_layered_decoder_cell,
                                                                         attention_mechanism=inference_attention_mechanism,
                                                                         attention_layer_size=cell_dimension)
            # create the initial state for the decoder
            inference_decoder_initial_state = inference_decoder_cell.zero_state(batch_size=tf.shape(input)[0],
                                                                                dtype=tf.float32).clone(
                cell_state=inference_encoder_state)
            inference_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(inputs=decoder_input,
                                                                                sequence_length=output_sequence_length,
                                                                                sampling_probability=1.0)
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=inference_decoder_cell, helper=inference_helper,
                                                                initial_state=inference_decoder_initial_state,
                                                                output_layer=dense_layer)

            # perform the decoding
            inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder)

        # error that should be minimized in the training process
        error = self.__l1_loss(training_outputs, training_targets)

        # l2 regularization of the trainable model parameters
        l2_loss = 0.0
        for var in tf.trainable_variables() :
            l2_loss += tf.nn.l2_loss(var)

        l2_loss = tf.multiply(tf.cast(l2_regularization, dtype=tf.float64), tf.cast(l2_loss, dtype=tf.float64))

        total_loss = tf.cast(error, dtype = tf.float64)+ l2_loss

        # create the optimizer
        optimizer = optimizer_fn(total_loss)

        # create the training and validation datasets from the tfrecord files
        training_dataset = tf.data.TFRecordDataset(filenames = [self.__binary_train_file_path], compression_type = "ZLIB")
        validation_dataset = tf.data.TFRecordDataset(filenames = [self.__binary_validation_file_path], compression_type = "ZLIB")

        # parse the records
        # parse the records
        moving_windowtfrecord_reader = MovingWindowTFRecordReader(self.__input_size, self.__output_size)
        non_moving_windowtfrecord_reader = NonMovingWindowTFRecordReader()

        # define the expected shapes of data after padding
        train_padded_shapes = ([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size])
        validation_padded_shapes = ([], [tf.Dimension(None), 1], [self.__output_size, 1], [1, 1])

        # preparing the training data
        shuffle_seed = tf.placeholder(dtype=tf.int64, shape=[])
        training_dataset = training_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=training_data_configs.SHUFFLE_BUFFER_SIZE,
                                                                  count=int(max_epoch_size), seed=shuffle_seed))
        training_dataset = training_dataset.map(moving_windowtfrecord_reader.train_data_parser)

        padded_training_data_batches = training_dataset.padded_batch(batch_size=int(minibatch_size),
                                                                     padded_shapes=train_padded_shapes)

        training_data_batch_iterator = padded_training_data_batches.make_initializable_iterator()
        next_training_data_batch = training_data_batch_iterator.get_next()

        # preparing the validation data
        validation_dataset = validation_dataset.map(non_moving_windowtfrecord_reader.validation_data_parser)

        # create a single batch from all the validation time series by padding the datasets to make the variable sequence lengths fixed
        padded_validation_dataset = validation_dataset.padded_batch(batch_size=int(minibatch_size),
                                                                    padded_shapes=validation_padded_shapes)

        # get an iterator to the validation data
        validation_data_iterator = padded_validation_dataset.make_initializable_iterator()

        # access the validation data using the iterator
        next_validation_data_batch = validation_data_iterator.get_next()

        # setup variable initialization
        init_op = tf.global_variables_initializer()

        # define the GPU options
        gpu_options = tf.GPUOptions(visible_device_list=gpu_configs.visible_device_list, allow_growth=True)

        with tf.Session(
                config=tf.ConfigProto(log_device_placement=gpu_configs.log_device_placement, allow_soft_placement=True,
                                      gpu_options=gpu_options)) as session:
            session.run(init_op)

            smape_final = 0.0
            smape_list = []
            for epoch in range(max_num_epochs):
                print("Epoch->", epoch)

                session.run(training_data_batch_iterator.initializer, feed_dict={shuffle_seed: epoch})

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
                            splitted_output = np.expand_dims(next_training_batch_value[2][:, i, :], axis=2)

                            # get the values of the input and output sequence lengths
                            length_comparison_array = np.greater([i + 1] * np.shape(splitted_input)[0],
                                                                 next_training_batch_value[0])
                            input_sequence_length_values = np.where(length_comparison_array, 0, self.__input_size)
                            output_sequence_length_values = np.where(length_comparison_array, 0, self.__output_size)

                            # define the teacher input signal for the decoder
                            decoder_input_value = np.hstack(
                                (np.expand_dims(splitted_input[:, -1, :], axis=1), splitted_output[:, :-1, :]))

                            # get the predictions
                            training_decoder_outputs_value, training_encoder_state_value = session.run(
                                [training_decoder_outputs, training_encoder_state], feed_dict={
                                    input: splitted_input,
                                    decoder_input: decoder_input_value,
                                    input_sequence_length: input_sequence_length_values,
                                    output_sequence_length: output_sequence_length_values,
                                    training_encoder_initial_state: training_encoder_state_value
                                })

                            # masking the outputs
                            training_predictions_value = np.ma.masked_array(
                                np.reshape(np.squeeze(training_decoder_outputs_value[0]),
                                           newshape=(actual_batch_size_value, self.__output_size)))
                            training_predictions_value[output_sequence_length_values == 0] = 0
                            training_predictions.append(training_predictions_value)

                        training_predictions = np.transpose(np.reshape(np.array(training_predictions), newshape=(np.shape(next_training_batch_value[1])[1], actual_batch_size_value, self.__output_size)), (1, 0, 2))
                        # backpropagate the accumulated errors
                        session.run(optimizer, feed_dict={
                            training_outputs: np.array(training_predictions),
                            training_targets: next_training_batch_value[2]
                        })
                    except tf.errors.OutOfRangeError:
                        break

            session.run(validation_data_iterator.initializer)
            while True:
                try:
                    # get the batch of validation inputs
                    validation_data_batch_value = session.run(next_validation_data_batch)
                    current_batch_size = np.shape(validation_data_batch_value[1])[0]

                    # shape for the target data
                    decoder_input_shape = [current_batch_size, self.__output_size, 1]

                    # apply the attention only to the last input_window size of the encoder outputs
                    inference_encoder_outputs_values = session.run(inference_encoder_outputs, feed_dict={
                        input: validation_data_batch_value[1],
                        input_sequence_length: validation_data_batch_value[0]
                    })

                    attention_memory_last_column_indices = validation_data_batch_value[0] - 1
                    attention_memory_start_column_indices = attention_memory_last_column_indices - self.__input_size + 1
                    attention_memory_column_indices = attention_memory_start_column_indices[:, None] + np.tile(
                        np.arange(self.__input_size), (current_batch_size, 1))

                    attention_memory_row_indices = np.tile(
                        np.reshape(np.arange(current_batch_size), (current_batch_size, -1)), (self.__input_size))
                    inference_attention_memory_values = inference_encoder_outputs_values[
                        attention_memory_row_indices, attention_memory_column_indices]

                    # get the output of the network for the validation input data batch
                    validation_output = session.run(inference_decoder_outputs[0],
                                                    feed_dict={input: validation_data_batch_value[1],
                                                               decoder_input: np.zeros(decoder_input_shape),
                                                               inference_attention_memory: inference_attention_memory_values,
                                                               input_sequence_length: validation_data_batch_value[0],
                                                               output_sequence_length: [self.__output_size] * current_batch_size
                                                               })

                    # calculate the smape for the validation data using vectorization

                    # convert the data to remove the preprocessing
                    level_values = validation_data_batch_value[3][:, 0, 0]

                    converted_validation_output = np.exp(
                        level_values[:, np.newaxis] + np.squeeze(validation_output, axis=2))

                    actual_values = validation_data_batch_value[2]
                    converted_actual_values = np.exp(
                        level_values[:, np.newaxis] + np.squeeze(actual_values, axis=2))

                    if (self.__contain_zero_values):  # to compensate for 0 values in data
                        converted_validation_output = converted_validation_output - 1
                        converted_actual_values = converted_actual_values - 1

                    if self.__integer_conversion:
                        converted_validation_output[converted_validation_output < 0] = 0
                        converted_validation_output = np.round(converted_validation_output)

                        converted_actual_values[converted_actual_values < 0] = 0
                        converted_actual_values = np.round(converted_actual_values)

                    if self.__address_near_zero_instability:
                        # calculate the smape
                        epsilon = 0.1
                        sum = np.maximum(np.abs(converted_validation_output) + np.abs(converted_actual_values) + epsilon, 0.5 + epsilon)
                        smape = np.mean(np.abs(converted_validation_output - converted_actual_values) /
                                        sum) * 2
                        smape_list.append(smape)
                    else:
                        # calculate the smape
                        smape = np.mean(np.abs(converted_validation_output - converted_actual_values) /
                                        (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
                        smape_list.append(smape)

                except tf.errors.OutOfRangeError:
                    break
            smape_final = np.mean(smape_list)
            print("SMAPE value: {}".format(smape_final))

        return smape_final



