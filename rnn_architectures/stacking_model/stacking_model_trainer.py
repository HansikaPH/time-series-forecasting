import numpy as np
import tensorflow as tf
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader
from configs.global_configs import model_training_configs
from configs.global_configs import training_data_configs
from configs.global_configs import gpu_configs

class StackingModelTrainer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__input_size = kwargs["input_size"]
        self.__output_size = kwargs["output_size"]
        self.__binary_train_file_path = kwargs["binary_train_file_path"]
        self.__binary_validation_file_path = kwargs["binary_validation_file_path"]
        self.__contain_zero_values = kwargs["contain_zero_values"]
        self.__address_near_zero_instability = kwargs["address_near_zero_instability"]
        self.__non_negative_integer_conversion = kwargs["non_negative_integer_conversion"]
        self.__seed = kwargs["seed"]
        self.__cell_type = kwargs["cell_type"]

    def __l1_loss(self, z, t):
        loss = tf.reduce_mean(tf.abs(t - z))
        return loss

    def __l2_loss(self, z, t):
        loss = tf.losses.mean_squared_error(labels=t, predictions=z)
        return loss

    # Training the time series
    def train_model(self, **kwargs):

        # extract the parameters from the kwargs
        num_hidden_layers = kwargs['num_hidden_layers']
        cell_dimension = kwargs['cell_dimension']
        minibatch_size = kwargs['minibatch_size']
        max_epoch_size = kwargs['max_epoch_size']
        max_num_epochs = kwargs['max_num_epochs']
        l2_regularization = kwargs['l2_regularization']
        gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        optimizer_fn = kwargs['optimizer_fn']
        random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']

        print(kwargs)

        tf.reset_default_graph()

        tf.set_random_seed(self.__seed)

        # declare the input and output placeholders

        # input format [batch_size, sequence_length, dimension]
        input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__input_size])
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
        training_input = input + noise

        validation_input = input

        # output format [batch_size, sequence_length, dimension]
        true_output = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])
        sequence_lengths = tf.placeholder(dtype=tf.int64, shape=[None])

        weight_initializer = tf.truncated_normal_initializer(stddev=random_normal_initializer_stdev)

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

        multi_layered_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell() for _ in range(int(num_hidden_layers))])

        with tf.variable_scope('train_scope') as train_scope:
            training_rnn_outputs, training_rnn_states = tf.nn.dynamic_rnn(cell=multi_layered_cell,
                                                                          inputs=training_input,
                                                                          sequence_length=sequence_lengths,
                                                                          dtype=tf.float32)

            # connect the dense layer to the RNN
            training_prediction_output = tf.layers.dense(
                inputs=tf.convert_to_tensor(value=training_rnn_outputs, dtype=tf.float32),
                units=self.__output_size,
                use_bias=self.__use_bias, kernel_initializer=weight_initializer, name='dense_layer')

        with tf.variable_scope(train_scope, reuse=tf.AUTO_REUSE) as inference_scope:
            inference_rnn_outputs, inference_rnn_states = tf.nn.dynamic_rnn(cell=multi_layered_cell,
                                                                            inputs=validation_input,
                                                                            sequence_length=sequence_lengths,
                                                                            dtype=tf.float32)
            # connect the dense layer to the RNN
            inference_prediction_output = tf.layers.dense(
                inputs=tf.convert_to_tensor(value=inference_rnn_outputs, dtype=tf.float32),
                units=self.__output_size,
                use_bias=self.__use_bias, kernel_initializer=weight_initializer, name='dense_layer', reuse=True)

        error = self.__l1_loss(training_prediction_output, true_output)

        # l2 regularization of the trainable model parameters
        l2_loss = 0.0
        for var in tf.trainable_variables():
            l2_loss += tf.nn.l2_loss(var)

        l2_loss = tf.multiply(tf.cast(l2_regularization, dtype=tf.float64), tf.cast(l2_loss, dtype=tf.float64))

        total_loss = tf.cast(error, dtype=tf.float64) + l2_loss

        # create the adagrad optimizer
        optimizer = optimizer_fn(total_loss)

        # create the training and validation datasets from the tfrecord files
        training_dataset = tf.data.TFRecordDataset(filenames=[self.__binary_train_file_path], compression_type="ZLIB")
        validation_dataset = tf.data.TFRecordDataset(filenames=[self.__binary_validation_file_path],
                                                     compression_type="ZLIB")

        # parse the records
        tfrecord_reader = TFRecordReader(self.__input_size, self.__output_size)

        # define the expected shapes of data after padding
        train_padded_shapes = ([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size])
        validation_padded_shapes = (
        [], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size],
        [tf.Dimension(None), self.__output_size + 1])

        # prepare the training data into batches
        # randomly shuffle the time series within the dataset and repeat for the value of the epoch size
        shuffle_seed = tf.placeholder(dtype=tf.int64, shape=[])
        # training_dataset = training_dataset.apply(
        #     tf.data.experimental.shuffle_and_repeat(buffer_size=training_data_configs.SHUFFLE_BUFFER_SIZE,
        #                                        count=int(max_epoch_size), seed=shuffle_seed))
        training_dataset = training_dataset.repeat(count=int(max_epoch_size))

        training_dataset = training_dataset.map(tfrecord_reader.train_data_parser)

        padded_training_data_batches = training_dataset.padded_batch(batch_size=int(minibatch_size),
                                                                     padded_shapes=train_padded_shapes)

        training_data_batch_iterator = padded_training_data_batches.make_initializable_iterator()
        next_training_data_batch = training_data_batch_iterator.get_next()

        # prepare the validation data into batches
        validation_dataset = validation_dataset.map(tfrecord_reader.validation_data_parser)

        # create a single batch from all the validation time series by padding the datasets to make the variable sequence lengths fixed
        padded_validation_dataset = validation_dataset.padded_batch(batch_size=minibatch_size,
                                                                    padded_shapes=validation_padded_shapes)

        # get an iterator to the validation data
        validation_data_iterator = padded_validation_dataset.make_initializable_iterator()

        # access the validation data using the iterator
        next_validation_data_batch = validation_data_iterator.get_next()

        # setup variable initialization
        init_op = tf.global_variables_initializer()

        # define the GPU options
        # gpu_options = tf.GPUOptions(visible_device_list=gpu_configs.visible_device_list, allow_growth=True)
        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(
                config=tf.ConfigProto(log_device_placement=gpu_configs.log_device_placement, allow_soft_placement=True,
                                      gpu_options=gpu_options)) as session:
            session.run(init_op)

            smape_final = 0.0
            smape_list = []
            for epoch in range(int(max_num_epochs)):
                print("Epoch->", epoch)

                session.run(training_data_batch_iterator.initializer, feed_dict={
                    shuffle_seed: epoch})  # initialize the iterator to the beginning of the training dataset

                while True:
                    try:
                        training_data_batch_value = session.run(next_training_data_batch,
                                                                feed_dict={shuffle_seed: epoch})

                        _, total_loss_value = session.run([optimizer, total_loss],
                                    feed_dict={training_input: training_data_batch_value[1],
                                               true_output: training_data_batch_value[2],
                                               sequence_lengths: training_data_batch_value[0]})

                    except tf.errors.OutOfRangeError:
                        break

            session.run(
                validation_data_iterator.initializer)  # initialize the iterator to the beginning of the training dataset

            while True:
                try:

                    # get the batch of validation inputs
                    validation_data_batch_value = session.run(next_validation_data_batch)

                    # get the output of the network for the validation input data batch
                    validation_output = session.run(inference_prediction_output,
                                                    feed_dict={input: validation_data_batch_value[1],
                                                               sequence_lengths: validation_data_batch_value[0]
                                                               })
                    # calculate the smape for the validation data using vectorization

                    # convert the data to remove the preprocessing
                    last_indices = validation_data_batch_value[0] - 1
                    array_first_dimension = np.array(range(0, validation_data_batch_value[0].shape[0]))

                    true_seasonality_values = validation_data_batch_value[3][array_first_dimension,
                                              last_indices, 1:]

                    level_values = validation_data_batch_value[3][array_first_dimension, last_indices, 0]

                    last_validation_outputs = validation_output[array_first_dimension, last_indices]
                    converted_validation_output = np.exp(
                        true_seasonality_values + level_values[:, np.newaxis] + last_validation_outputs)

                    actual_values = validation_data_batch_value[2][array_first_dimension, last_indices, :]
                    converted_actual_values = np.exp(
                        true_seasonality_values + level_values[:, np.newaxis] + actual_values)

                    if (self.__contain_zero_values):  # to compensate for 0 values in data
                        converted_validation_output = converted_validation_output - 1
                        converted_actual_values = converted_actual_values - 1

                    if self.__non_negative_integer_conversion:
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
            session.close()

        return smape_final
