import numpy as np
import pandas as pd
import random
import csv

import tensorflow as tf

# Input/Output Window size.
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Training and Validation file paths.
train_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/stl_12i15.txt'
validate_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/stl_12i15v.txt'
test_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/cif12test.txt'

# global lists for storing the data from files
list_of_trainig_inputs = []
list_of_training_labels = []
list_of_levels = []
list_of_true_values = []
list_of_true_seasonality = []
list_of_test_inputs = []


# TODO: move the common parts of the evaluation and training script to separate script
# TODO: see how to get the same trained model in the training script to here as well
# TODO: issue with

def L1Loss(z, t):
    loss = tf.reduce_mean(tf.abs(t - z))
    return loss

# Preparing training dataset.
def create_train_data():

    # Reading the training dataset.
    train_df = pd.read_csv(train_file_path, nrows=10)

    float_cols = [c for c in train_df if train_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    train_df = pd.read_csv(train_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    train_df = train_df.rename(columns={0: 'series'})

    # Returns unique number of time series in the dataset.
    series = np.unique(train_df['series'])

    # Construct input and output training tuples for each time series.
    for ser in series:
        oneSeries_df = train_df[train_df['series'] == ser]
        inputs_df = oneSeries_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        labels_df = oneSeries_df.iloc[:, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
        list_of_trainig_inputs.append(np.ascontiguousarray(inputs_df, dtype=np.float32))
        list_of_training_labels.append(np.ascontiguousarray(labels_df, dtype=np.float32))

    # # Reading the validation dataset.
    # val_df = pd.read_csv(validate_file_path, nrows=10)
    #
    # float_cols = [c for c in val_df if val_df[c].dtype == "float64"]
    # float32_cols = {c: np.float32 for c in float_cols}
    #
    # val_df = pd.read_csv(validate_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)
    #
    # val_df = val_df.rename(columns={0: 'series'})
    # val_df = val_df.rename(columns={(INPUT_SIZE + OUTPUT_SIZE + 3): 'level'})
    # series = np.unique(val_df['series'])
    #
    # for ser in series:
    #     oneSeries_df = val_df[val_df['series'] == ser]
    #     inputs_df_test = oneSeries_df.iloc[:, range(1, (INPUT_SIZE + 1))]
    #     labels_df_test = oneSeries_df.iloc[:, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
    #     level = np.ascontiguousarray(oneSeries_df['level'], dtype=np.float32)
    #     level = level[level.shape[0] - 1]
    #     trueValues_df = oneSeries_df.iloc[
    #         oneSeries_df.shape[0] - 1, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
    #     trueSeasonailty_df = oneSeries_df.iloc[
    #         oneSeries_df.shape[0] - 1, range((INPUT_SIZE + OUTPUT_SIZE + 4), oneSeries_df.shape[1])]
    #     list_of_validation_inputs.append(np.ascontiguousarray(inputs_df_test, dtype=np.float32))
    #     list_of_validation_labels.append(np.ascontiguousarray(labels_df_test, dtype=np.float32))
    #     list_of_levels.append(level)
    #     list_of_true_values.append(np.ascontiguousarray(trueValues_df, dtype=np.float32))
    #     list_of_true_seasonality.append(np.ascontiguousarray(trueSeasonailty_df, dtype=np.float32))

    # Reading the test file.
    test_df = pd.read_csv(test_file_path, nrows=10)

    float_cols = [c for c in test_df if test_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    test_df = pd.read_csv(test_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    test_df = test_df.rename(columns={0: 'series'})

    series1 = np.unique(test_df['series'])

    for ser in series1:
        test_series_df = test_df[test_df['series'] == ser]
        test_inputs_df = test_series_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        list_of_test_inputs.append(np.ascontiguousarray(test_inputs_df, dtype=np.float32))


# returns a list of the lengths of the sequences in a batch
def length(batch):
  used = tf.sign(tf.reduce_max(tf.abs(batch), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

# Training the time series
def train_model():

    # optimized hyperparameters
    maxNumOfEpochs =20
    maxEpochSize = 1
    learningRate = 0.0013262220421187676
    lstmCellDimension = 23
    l2_regularization = 0.00015753660121731034
    mbSize =10

    # reset the tensorflow graph
    tf.reset_default_graph()

    # declare the input and output placeholders
    input = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
    label = tf.placeholder(dtype=tf.float32, shape=[None, None, OUTPUT_SIZE])

    # create the model architecture

    # RNN with the LSTM layer
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(lstmCellDimension),
                                        use_peepholes=LSTM_USE_PEEPHOLES)  # TODO: self stabilization - not quite needed

    rnn_outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input, sequence_length=length(input),
                                            dtype=tf.float32)

    # connect the dense layer to the RNN
    dense_layer = tf.layers.dense(inputs=tf.convert_to_tensor(value=rnn_outputs, dtype=tf.float32), units=OUTPUT_SIZE,
                                  use_bias=BIAS)

    # error that should be minimized the training process
    error = L1Loss(dense_layer, label)

    # l2 regularization of the trainable model parameters
    l2_loss = 0.0
    for var in tf.trainable_variables():
        l2_loss += tf.nn.l2_loss(var)

    l2_loss = tf.multiply(l2_regularization, l2_loss)

    total_loss = error + l2_loss

    # create the adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate=learningRate).minimize(
        total_loss)  # TODO: gaussian_noise_injection_std_dev=gaussianNoise

    # create the Dataset objects for the training input and label data
    training_inputs = tf.data.Dataset.from_generator(generator=lambda: list_of_trainig_inputs, output_types=tf.float32)
    training_labels = tf.data.Dataset.from_generator(generator=lambda: list_of_training_labels, output_types=tf.float32)

    # zip the training input and label datasets together
    training_dataset = tf.data.Dataset.zip((training_inputs, training_labels))

    # create the Dataset objects for the test input data with padded values
    test_dataset = tf.data.Dataset.from_generator(generator=lambda: list_of_test_inputs,
                                                       output_types=tf.float32)
    # validation_labels = tf.data.Dataset.from_generator(generator=lambda: list_of_validation_labels,
    #                                                    output_types=tf.float32)

    # zip the validation input and label data
    # validation_input_label_dataset = tf.data.Dataset.zip((test_inputs, validation_labels))

    # create Dataset objects for the validation levels, true values and true seasonality
    # validation_levels = tf.data.Dataset.from_generator(generator=lambda: list_of_levels, output_types=tf.float32)
    # validation_true_values = tf.data.Dataset.from_generator(generator=lambda: list_of_true_values,
    #                                                         output_types=tf.float32)
    # validation_true_seasonality = tf.data.Dataset.from_generator(generator=lambda: list_of_true_seasonality,
    #                                                              output_types=tf.float32)

    # zip the validation datasets except the inputs and labels
    # validation_dataset = tf.data.Dataset.zip((validation_levels, validation_true_values, validation_true_seasonality))

    INFO_FREQ = 1
    # sMAPE_final_list = []

    # setup variable initialization
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)

        for iscan in range(int(maxNumOfEpochs)):
            # sMAPE_epoch_list = []
            print("Epoch->", iscan)

            # randomly shuffle the time series within the dataset
            training_dataset.shuffle(mbSize)

            for epochsize in range(int(maxEpochSize)):
                # sMAPE_list = []

                # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
                padded_training_data_batches = training_dataset.padded_batch(batch_size = int(mbSize),
                                                                             padded_shapes = ([tf.Dimension(None), INPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE]))

                # get an iterator to the batches
                training_data_batch_iterator = padded_training_data_batches.make_one_shot_iterator()

                # access each batch using the iterator
                next_training_data_batch = training_data_batch_iterator.get_next()

                while True:
                    try:
                        next_training_batch_value = session.run(next_training_data_batch)

                        # model training
                        session.run(optimizer,
                                    feed_dict={input: next_training_batch_value[0],
                                               label: next_training_batch_value[1]})
                    except tf.errors.OutOfRangeError:
                        break

        # applying the model to the test data

        # create a single batch from all the test time series by padding the datasets to make the variable sequence lengths fixed
        padded_test_input_data = test_dataset.padded_batch(batch_size=len(list_of_test_inputs), padded_shapes = ([tf.Dimension(None), INPUT_SIZE]))

        # # create a single batches for the rest of the validation data without padding
        # validation_data_batches = validation_dataset.batch(batch_size=len(list_of_levels))

        # get an iterator to the test input data batch
        test_input_iterator = padded_test_input_data.make_one_shot_iterator()

        # get an iterator to the validation data batch
        # validation_data_batch_iterator = validation_data_batches.make_one_shot_iterator()

        # access the test input batch using the iterator
        test_input_data_batch = test_input_iterator.get_next()

        # access the validation data batch using the iterator
        # validation_data_batch = validation_data_batch_iterator.get_next()

        # get the batch of test inputs
        test_input_batch_value = session.run(test_input_data_batch)

        # get the next batch of the other validation data
        # validation_data_batch_value = session.run(validation_data_batch)

        # get the output of the network for the validation input data batch
        test_output = session.run(dense_layer,
                                        feed_dict={input: test_input_batch_value})

        # get the lengths of the output data
        output_sequence_lengths = session.run(length(test_output))

        # iterate across the different time series
        # for i in range(test_output.shape[0]):
        #     # convert the data to remove the preprocessing
        #     true_seasonality_values = validation_data_batch_value[2][i]
        #     level = validation_data_batch_value[1][i]
        #     last_index = output_sequence_lengths[i] - 1
        #     last_validation_output = test_output[i, last_index]
        #
        #     converted_validation_output = true_seasonality_values + level + last_validation_output
        #     actual_values = test_input_batch_value[1][i, last_index]
        #     converted_actual_values = actual_values + true_seasonality_values + level

            # # calculate the smape
            # sMAPE = np.mean(np.abs(converted_validation_output - converted_actual_values) /
            #                 (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
            # sMAPE_list.append(sMAPE)

#     sMAPEprint = np.mean(sMAPE_list)
#     sMAPE_epoch_list.append(sMAPEprint)
#
# sMAPE_validation = np.mean(sMAPE_epoch_list)
# sMAPE_final_list.append(sMAPE_validation)
# print("smape: ", sMAPE_validation)
#
# sMAPE_final = np.mean(sMAPE_final_list)
# max_value = 1 / (sMAPE_final)

        # print(test_output)
        # print(np.shape(test_output))
        return test_output, output_sequence_lengths


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)

    create_train_data()

    test_output, output_sequence_lengths = train_model()

    list_of_forecasts = []

    # Writes the last test output(i.e Forecast) of each time series to a file
    for time_series in range(len(test_output)):
        last_output_index = output_sequence_lengths[time_series] - 1
        forecast = test_output[time_series][last_output_index]
        list_of_forecasts.append(forecast)

    with open("Forecasts/forecasts.txt", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_forecasts)
