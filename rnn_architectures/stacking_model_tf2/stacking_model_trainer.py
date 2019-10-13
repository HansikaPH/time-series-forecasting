from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tfrecords_handler.moving_window.tfrecord_reader import TFRecordReader
from configs.global_configs import model_training_configs
from configs.global_configs import training_data_configs
from configs.global_configs import gpu_configs
from external_packages.cocob_optimizer import cocob_optimizer

# how to bucket by the sequence lengths
# reuse the datasets
# make the code cleaner and remove duplicate code
# define custom loss
# tensorboard

class StackingModelTrainer:

    def __init__(self, **kwargs):
        self.__use_bias = kwargs["use_bias"]
        self.__use_peepholes = kwargs["use_peepholes"]
        self.__input_size = kwargs["input_size"]
        self.__output_size = kwargs["output_size"]
        self.__binary_train_file_path = kwargs["binary_train_file_path"]
        self.__binary_validation_file_path = kwargs["binary_validation_file_path"]
        self.__binary_test_file_path = kwargs["binary_test_file_path"]
        self.__contain_zero_values = kwargs["contain_zero_values"]
        self.__address_near_zero_instability = kwargs["address_near_zero_instability"]
        self.__integer_conversion = kwargs["integer_conversion"]
        self.__seed = kwargs["seed"]
        self.__cell_type = kwargs["cell_type"]
        self.__without_stl_decomposition = kwargs["without_stl_decomposition"]

        # define the metadata size based on the usage of stl decomposition
        if self.__without_stl_decomposition:
            self.__meta_data_size = 1
        else:
            self.__meta_data_size = self.__output_size + 1

    def __create_training_datasets(self):
        self.__training_dataset_for_train = tf.data.TFRecordDataset(filenames=[self.__binary_train_file_path], compression_type="ZLIB")
        self.__validation_dataset_for_train = tf.data.TFRecordDataset(filenames=[self.__binary_validation_file_path],
                                                     compression_type="ZLIB")

        # parse the records
        self.__tfrecord_reader = TFRecordReader(self.__input_size, self.__output_size, self.__meta_data_size)

        # define the expected shapes of data after padding
        train_padded_shapes = ([None, self.__input_size], [None, self.__output_size])
        validation_padded_shapes = ([None, self.__input_size])

        self.__training_dataset_for_train = self.__training_dataset_for_train.map(self.__tfrecord_reader.train_data_parser_2)
        self.__validation_dataset_for_train_2 = self.__validation_dataset_for_train.map(self.__tfrecord_reader.validation_data_parser_2)

        # find the size of the dataset - take as an external argument later
        self.__no_of_sequences = 0
        for sequence in self.__validation_dataset_for_train:
            self.__no_of_sequences += 1
        self.__validation_dataset_for_train_2 = self.__validation_dataset_for_train_2.padded_batch(batch_size=int(self.__no_of_sequences),
                                                                 padded_shapes=validation_padded_shapes)
        self.__validation_lengths = self.__validation_dataset_for_train.map(self.__tfrecord_reader.validation_data_parser_3)
        self.__validation_lengths = self.__validation_lengths.batch(batch_size=int(self.__no_of_sequences))
        self.__validation_metadata = self.__validation_dataset_for_train.map(self.__tfrecord_reader.validation_data_parser_4)
        self.__validation_metadata = self.__validation_metadata.padded_batch(batch_size=int(self.__no_of_sequences),
                                                               padded_shapes=[None, self.__meta_data_size])
        self.__validation_actuals = self.__validation_dataset_for_train.map(self.__tfrecord_reader.validation_data_parser_5)
        self.__validation_actuals = self.__validation_actuals.padded_batch(batch_size=int(self.__no_of_sequences),
                                                             padded_shapes=[None, self.__output_size])

        for batch1, batch2, batch3 in zip(self.__validation_lengths, self.__validation_metadata, self.__validation_actuals):
            self.__validation_lengths = batch1.numpy()
            self.__validation_metadata = batch2.numpy()
            self.__validation_actuals = batch3.numpy()

        self.__last_indices = self.__validation_lengths - 1
        self.__array_first_dimension = np.array(range(0, self.__no_of_sequences))

        self.__true_seasonality_values = self.__validation_metadata[self.__array_first_dimension,
                                  self.__last_indices, 1:]

        self.__level_values = self.__validation_metadata[self.__array_first_dimension, self.__last_indices, 0]

    def __create_testing_datasets(self):
        return
    def __get_optimizer(self, optimizer, initial_learning_rate):
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = initial_learning_rate)
        elif optimizer == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=initial_learning_rate)
        elif optimizer == "cocob":
            optimizer = cocob_optimizer.COCOB()
        return optimizer

    def build_model(self, **kwargs):

        num_hidden_layers = int(kwargs['num_hidden_layers'])
        cell_dimension = int(kwargs['cell_dimension'])
        minibatch_size = int(kwargs['minibatch_size'])
        max_epoch_size = int(kwargs['max_epoch_size'])
        max_num_epochs = int(kwargs['max_num_epochs'])
        l2_regularization = kwargs['l2_regularization']
        gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        optimizer = kwargs['optimizer']
        # initial_learning_rate = kwargs['initial_learning_rate']
        random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']

        self.__create_training_datasets()

        tf.keras.backend.clear_session()
        tf.compat.v1.random.set_random_seed(self.__seed)

        train_padded_shapes = ([None, self.__input_size], [None, self.__output_size])
        validation_padded_shapes = ([None, self.__input_size])

        train_dataset = self.__training_dataset_for_train.repeat(max_epoch_size)
        train_dataset = train_dataset.padded_batch(batch_size=int(minibatch_size), padded_shapes=train_padded_shapes)

        initializer = tf.keras.initializers.TruncatedNormal(stddev=random_normal_initializer_stdev)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Masking(mask_value = 0.0, input_shape=(None, self.__input_size)))

        # for normal lstm cells
        # for i in range(num_hidden_layers):
        #     lstm_layer = tf.keras.layers.LSTM(cell_dimension, kernel_initializer = initializer, return_sequences=True)
        #     model.add(lstm_layer)

        # for lstm cells with peephole connections
        lstm_stacks = []
        for layers in range(num_hidden_layers):
            lstm_stacks.append(tf.keras.experimental.PeepholeLSTMCell(cell_dimension,
                                                                      kernel_initializer=initializer
                                                                      ))

        # create RNN layer from lstm stacks
        rnn_layer = tf.keras.layers.RNN(lstm_stacks, return_sequences=True)
        model.add(rnn_layer)

        # model.add(tf.keras.layers.BatchNormalization())
        model.add( tf.keras.layers.Dense(self.__output_size, use_bias = self.__use_bias, kernel_initializer=initializer))

        def custom_mae(y_true, y_pred):
            error = tf.keras.losses.mae(y_true, y_pred)
            l2_loss = 0.0
            for var in model.trainable_variables:
                l2_loss += tf.nn.l2_loss(var)

            l2_loss = tf.math.multiply(l2_regularization, l2_loss)

            total_loss = error + l2_loss
            return total_loss

        # training
        model.compile(loss=custom_mae,
                      optimizer=self.__get_optimizer(optimizer, 0.0),
                      metrics=['mean_absolute_error'])
        # # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True,
        # #                                             mode='min')
        # # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=7, mode='min',
        # #                                                  cooldown=15)
        history = model.fit(train_dataset, epochs = max_num_epochs
                            # , shuffle=True
                            # ,callbacks=[callback, reduce_lr]
        )

        # validation
        validation_result = model.predict(validation_dataset_2)
        # calculate the validation losses

        # last_indices = self.__validation_lengths - 1
        # array_first_dimension = np.array(range(0, self.__no_of_sequences))
        #
        # true_seasonality_values = self.__validation_metadata[array_first_dimension,
        #                           last_indices, 1:]
        #
        # level_values = metadata[array_first_dimension, last_indices, 0]
        #
        # last_validation_outputs = validation_result[self.__array_first_dimension, self.__last_indices]
        # actual_values = self.__validation_actuals[self.__array_first_dimension, self.__last_indices, :]
        #
        #
        # if self.__without_stl_decomposition:
        #     converted_validation_output = np.exp(last_validation_outputs)
        #     converted_actual_values = np.exp(actual_values)
        #
        # else:
        #     converted_validation_output = np.exp(
        #         self.__true_seasonality_values + self.__level_values[:, np.newaxis] + last_validation_outputs)
        #     converted_actual_values = np.exp(self.__true_seasonality_values + self.__level_values[:, np.newaxis] + actual_values)
        #
        # if self.__contain_zero_values:  # to compensate for 0 values in data
        #     converted_validation_output = converted_validation_output - 1
        #     converted_actual_values = converted_actual_values - 1
        #
        # if self.__without_stl_decomposition:
        #     converted_validation_output = converted_validation_output * self.__level_values[:, np.newaxis]
        #     converted_actual_values = converted_actual_values * self.__level_values[:, np.newaxis]
        #
        # if self.__integer_conversion:
        #     converted_validation_output = np.round(converted_validation_output)
        #     converted_actual_values = np.round(converted_actual_values)
        #
        # converted_validation_output[converted_validation_output < 0] = 0
        # converted_actual_values[converted_actual_values < 0] = 0
        #
        # if self.__address_near_zero_instability:
        #     # calculate the smape
        #     epsilon = 0.1
        #     sum = np.maximum(np.abs(converted_validation_output) + np.abs(converted_actual_values) + epsilon,
        #                      0.5 + epsilon)
        #     smape_values = (np.abs(converted_validation_output - converted_actual_values) /
        #                     sum) * 2
        #     smape_values_per_series = np.mean(smape_values, axis=1)
        # else:
        #     # calculate the smape
        #     smape_values = (np.abs(converted_validation_output - converted_actual_values) /
        #                     (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
        #     smape_values_per_series = np.mean(smape_values, axis=1)
        #
        # smape = np.mean(smape_values_per_series)

        return 2.0

    def test_model(self, kwargs, seed):

        num_hidden_layers = int(kwargs['num_hidden_layers'])
        cell_dimension = int(kwargs['cell_dimension'])
        minibatch_size = int(kwargs['minibatch_size'])
        max_epoch_size = int(kwargs['max_epoch_size'])
        max_num_epochs = int(kwargs['max_num_epochs'])
        l2_regularization = kwargs['l2_regularization']
        gaussian_noise_stdev = kwargs['gaussian_noise_stdev']
        optimizer = kwargs['optimizer']
        # initial_learning_rate = kwargs['initial_learning_rate']
        random_normal_initializer_stdev = kwargs['random_normal_initializer_stdev']

        tf.keras.backend.clear_session()
        tf.compat.v1.random.set_random_seed(self.__seed)

        # prepare the data
        # training_dataset = tf.data.TFRecordDataset(filenames=[self.__binary_train_file_path], compression_type="ZLIB")
        validation_dataset = tf.data.TFRecordDataset(filenames=[self.__binary_validation_file_path],
                                                     compression_type="ZLIB")
        test_dataset = tf.data.TFRecordDataset(filenames=[self.__binary_test_file_path],
                                                     compression_type="ZLIB")

        # parse the records
        tfrecord_reader = TFRecordReader(self.__input_size, self.__output_size, self.__meta_data_size)

        # define the expected shapes of data after padding
        test_padded_shapes = ([None, self.__input_size])
        validation_padded_shapes = ([None, self.__input_size], [None, self.__output_size])

        # training_dataset = training_dataset.map(tfrecord_reader.train_data_parser_2)
        validation_dataset_train = validation_dataset.map(tfrecord_reader.validation_data_parser_6)
        validation_dataset_train = validation_dataset_train.repeat(max_epoch_size)

        testing_lengths = test_dataset.map(tfrecord_reader.test_data_parser_2)
        test_dataset = test_dataset.map(tfrecord_reader.test_data_parser)


        # find the size of the dataset - take as an external argument later
        no_of_sequences = 0
        for sequence in validation_dataset:
            no_of_sequences += 1


        testing_lengths = testing_lengths.batch(batch_size=no_of_sequences)
        for batch1 in testing_lengths:
            testing_lengths = batch1.numpy()

        test_dataset = test_dataset.padded_batch(batch_size=int(no_of_sequences),
                                                                 padded_shapes=test_padded_shapes)

        validation_dataset_train = validation_dataset_train.padded_batch(batch_size=int(minibatch_size),
                                                 padded_shapes=validation_padded_shapes)

        # regularizer = tf.keras.regularizers.l2(l = l2_regularization)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Masking(mask_value = 0.0, input_shape=(None, self.__input_size)))

        initializer = tf.keras.initializers.TruncatedNormal( stddev=random_normal_initializer_stdev)
        lstm_stacks = []
        for layers in range(num_hidden_layers):
            lstm_stacks.append(tf.keras.experimental.PeepholeLSTMCell(cell_dimension, kernel_initializer=initializer))

        # create RNN layer from lstm stacks
        rnn_layer = tf.keras.layers.RNN(lstm_stacks, return_sequences = True)
        model.add(rnn_layer)

        # for i in range(num_hidden_layers):
        #     lstm_layer = tf.keras.layers.LSTM(cell_dimension, kernel_initializer=initializer, return_sequences=True)
        #     model.add(lstm_layer)

        # model.add(tf.keras.layers.BatchNormalization(beta_regularizer = regularizer, gamma_regularizer = regularizer))
        model.add(
            tf.keras.layers.Dense(self.__output_size, use_bias=self.__use_bias, kernel_initializer=initializer
                                  # ,kernel_regularizer = regularizer, bias_regularizer = regularizer, activity_regularizer = regularizer
                                  ))

        # tf.keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)

        def mae(y_true, y_pred):
            error = tf.keras.losses.mae(y_true, y_pred)
            l2_loss = 0.0
            for var in model.trainable_variables:
                l2_loss += tf.nn.l2_loss(var)

            l2_loss = tf.math.multiply(l2_regularization, l2_loss)

            total_loss = error + l2_loss
            return total_loss

        # training
        model.compile(loss=mae,
                      optimizer=self.__get_optimizer(optimizer, 0.0),
                      metrics=['mean_squared_error'])
        # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True,
        #                                             mode='min')
        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=7, mode='min',
        #                                                  cooldown=15)
        history = model.fit(validation_dataset_train, epochs=max_num_epochs
                            # , shuffle=True
                            # ,callbacks=[callback, reduce_lr]
                            )

        # testing
        test_result = model.predict(test_dataset)

        # extracting the final time step forecast
        last_output_index = testing_lengths - 1
        array_first_dimension = np.array(range(0, no_of_sequences))
        forecasts = test_result[array_first_dimension, last_output_index]
        return forecasts