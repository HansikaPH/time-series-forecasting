import numpy as np
import tensorflow as tf
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader
from configs.global_configs import training_data_configs
from configs.global_configs import gpu_configs

class Seq2SeqModelTesterWithDenseLayer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__input_size = kwargs["input_size"]
        self.__output_size = kwargs["output_size"]
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
        input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__input_size])
        testing_input = input
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=gaussian_noise_stdev, dtype=tf.float32)
        training_input = input + noise

        target = tf.placeholder(dtype=tf.float32, shape=[None, None, self.__output_size])

        # placeholder for the sequence lengths
        sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        weight_initializer = tf.truncated_normal_initializer(stddev=random_normal_initializer_stdev)

        # create a tensor array for the indices of the encoder outputs array and the target
        new_index_array = tf.range(start=0, limit=tf.shape(sequence_length)[0], delta=1)
        output_array_indices = tf.stack([new_index_array, sequence_length - 1], axis=-1)

        actual_targets = tf.gather_nd(params=target, indices=output_array_indices)
        actual_targets = tf.expand_dims(input=actual_targets, axis=1)

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

        # building the decoder network for training
        with tf.variable_scope('dense_layer_train_scope') as dense_layer_train_scope:
            train_final_timestep_predictions = tf.gather_nd(params=training_encoder_outputs, indices=output_array_indices)

            # the final projection layer to convert the encoder_outputs to the desired dimension
            train_prediction_output = tf.layers.dense(
                inputs=tf.convert_to_tensor(value=train_final_timestep_predictions, dtype=tf.float32),
                units=self.__output_size,
                use_bias=self.__use_bias, kernel_initializer=weight_initializer)
            train_prediction_output = tf.expand_dims(input=train_prediction_output, axis=1)

        # building the decoder network for inference
        with tf.variable_scope(dense_layer_train_scope, reuse=tf.AUTO_REUSE) as dense_layer_inference_scope:
            inference_final_timestep_predictions = tf.gather_nd(params=inference_encoder_outputs,
                                                            indices=output_array_indices)

            # the final projection layer to convert the encoder_outputs to the desired dimension
            inference_prediction_output = tf.layers.dense(
                inputs=tf.convert_to_tensor(value=inference_final_timestep_predictions, dtype=tf.float32),
                units=self.__output_size,
                use_bias=self.__use_bias, kernel_initializer=weight_initializer)
            inference_prediction_output = tf.expand_dims(input=inference_prediction_output, axis=1)

        # error that should be minimized in the training process
        error = self.__l1_loss(train_prediction_output, actual_targets)

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

        # preparing the training data
        # randomly shuffle the time series within the dataset
        shuffle_seed = tf.placeholder(dtype=tf.int64, shape=[])
        # training_dataset = training_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=training_data_configs.SHUFFLE_BUFFER_SIZE,
        #                                                           count=int(max_epoch_size), seed=shuffle_seed))
        training_dataset = training_dataset.repeat(count=int(max_epoch_size))
        training_dataset = training_dataset.map(tfrecord_reader.validation_data_parser)

        # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
        padded_training_data_batches = training_dataset.padded_batch(batch_size=int(minibatch_size),
                                                                     padded_shapes=([], [tf.Dimension(None), self.__input_size], [tf.Dimension(None), self.__output_size],
                                                                                    [tf.Dimension(None), self.__output_size + 1]))

        # get an iterator to the batches
        training_data_batch_iterator = padded_training_data_batches.make_initializable_iterator()

        # access each batch using the iterator
        next_training_data_batch = training_data_batch_iterator.get_next()

        # preparing the test data
        test_dataset = test_dataset.map(tfrecord_reader.test_data_parser)

        # create a single batch from all the test time series by padding the datasets to make the variable sequence lengths fixed
        padded_test_input_data = test_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes=(
        [], [tf.Dimension(None),self.__input_size], [tf.Dimension(None), self.__output_size + 1]))

        # get an iterator to the test input data batch
        test_input_iterator = padded_test_input_data.make_one_shot_iterator()

        # access the test input batch using the iterator
        test_input_data_batch = test_input_iterator.get_next()

        # setup variable initialization
        init_op = tf.global_variables_initializer()
        #
        # writer_train = tf.summary.FileWriter('./logs/plot_train')
        # loss_var = tf.Variable(0.0)
        # tf.summary.scalar("loss", loss_var)
        # write_op = tf.summary.merge_all()

        # define the GPU options
        # gpu_options = tf.GPUOptions(visible_device_list=gpu_configs.visible_device_list, allow_growth=True)
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(
                config=tf.ConfigProto(log_device_placement=gpu_configs.log_device_placement, allow_soft_placement=True,
                                      gpu_options=gpu_options)) as session:
            session.run(init_op)

            for epoch in range(int(max_num_epochs)):
                print("Epoch->", epoch)

                session.run(training_data_batch_iterator.initializer, feed_dict={shuffle_seed: epoch})
                losses = []
                while True:
                    try:
                        next_training_batch_value = session.run(next_training_data_batch, feed_dict={shuffle_seed: epoch})

                        # model training
                        _, loss_val = session.run([optimizer, total_loss],
                                    feed_dict={input: next_training_batch_value[1],
                                               target: next_training_batch_value[2],
                                               sequence_length: next_training_batch_value[0],
                                               })
                        losses.append(loss_val)
                    except tf.errors.OutOfRangeError:
                        break
                # summary = session.run(write_op, {loss_var: np.mean(losses)})
                # writer_train.add_summary(summary, epoch)
                # writer_train.flush()

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

            return np.squeeze(list_of_forecasts, axis = 1) #the second dimension is squeezed since it is one