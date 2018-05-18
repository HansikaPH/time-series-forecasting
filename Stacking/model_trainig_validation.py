import numpy as np
import pandas as pd
import random

from bayes_opt import BayesianOptimization

import tensorflow as tf

# TODO: init_points in bayesian optimization

# TODO: see the tensorflow way of batching the input data - the fifo queues etc...
# ** TODO: understand the length calculation step of the sequences - see better way to get the length(using protocol buffers)
# ** TODO: make the code more efficient
# TODO: get the results and keep improving, run the cntk code and compare the results
# TODO: make the code proper(variable names, what variables to actually create)
# ** TODO: see if the matrix, tuple indices are correct, see if the code is correct as a whole
# ** TODO: see about integrating the progress printer
# TODO: see about resetting the graph on each iteration of bayesian optimization
# TODO: add l2 regularization and gaussian noise in training

# Input/Output Window sizes
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Training and Validation file paths.
# TODO: change the way the file paths are taken
train_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/stl_12i15.txt'
validate_file_path = '/home/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/stl_12i15v.txt'

# global lists for storing the data from files
list_of_trainig_inputs = []
list_of_training_labels = []
list_of_validation_inputs = []
list_of_validation_labels = []
list_of_levels = []
list_of_true_values = []
list_of_true_seasonality = []

def L1Loss(z, t):
    loss = tf.reduce_mean(tf.abs(t - z))
    return loss

# Preparing training dataset.
def create_train_data():

    # Reading the training dataset.
    train_df = pd.read_csv(train_file_path, nrows=10)

    float_cols = [c for c in train_df if train_df[c].dtype == "float64"] # why change float64 to float32?
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

    # Reading the validation dataset.
    val_df = pd.read_csv(validate_file_path, nrows=10)

    float_cols = [c for c in val_df if val_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    val_df = pd.read_csv(validate_file_path, sep=" ", header=None, engine='c', dtype=float32_cols)

    val_df = val_df.rename(columns={0: 'series'})
    val_df = val_df.rename(columns={(INPUT_SIZE + OUTPUT_SIZE + 3): 'level'})
    series = np.unique(val_df['series'])

    for ser in series:
        oneSeries_df = val_df[val_df['series'] == ser]
        inputs_df_test = oneSeries_df.iloc[:, range(1, (INPUT_SIZE + 1))]
        labels_df_test = oneSeries_df.iloc[:, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
        level = np.ascontiguousarray(oneSeries_df['level'], dtype=np.float32)
        level = level[level.shape[0] - 1]
        trueValues_df = oneSeries_df.iloc[
            oneSeries_df.shape[0] - 1, range((INPUT_SIZE + 2), (INPUT_SIZE + OUTPUT_SIZE + 2))]
        trueSeasonailty_df = oneSeries_df.iloc[
            oneSeries_df.shape[0] - 1, range((INPUT_SIZE + OUTPUT_SIZE + 4), oneSeries_df.shape[1])]
        list_of_validation_inputs.append(np.ascontiguousarray(inputs_df_test, dtype=np.float32))
        list_of_validation_labels.append(np.ascontiguousarray(labels_df_test, dtype=np.float32))
        list_of_levels.append(level)
        list_of_true_values.append(np.ascontiguousarray(trueValues_df, dtype=np.float32))
        list_of_true_seasonality.append(np.ascontiguousarray(trueSeasonailty_df, dtype=np.float32))

# returns a list of the lengths of the sequences in a batch
def length(batch):
  used = tf.sign(tf.reduce_max(tf.abs(batch), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

# Training the time series
def train_model(learningRate, lstmCellDimension, mbSize, maxEpochSize, maxNumOfEpochs,l2_regularization):

    tf.reset_default_graph()

    # declare the input and output placeholders
    input = tf.placeholder(dtype = tf.float32, shape = [None, None, INPUT_SIZE])
    label = tf.placeholder(dtype = tf.float32, shape = [None, None, OUTPUT_SIZE])

    # create the model architecture

    # RNN with the LSTM layer
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = int(lstmCellDimension), use_peepholes = LSTM_USE_PEEPHOLES) # TODO: self stabilization - not quite needed
    rnn_outputs, states = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = input, sequence_length = length(input), dtype = tf.float32)

    # connect the dense layer to the RNN
    dense_layer = tf.layers.dense(inputs = tf.convert_to_tensor(value = rnn_outputs, dtype = tf.float32), units = OUTPUT_SIZE, use_bias = BIAS)

    # error that should be minimized the training process
    error = L1Loss(dense_layer, label)

    # l2 regularization of the trainable model parameters
    l2_loss = 0.0
    for var in tf.trainable_variables() :
        l2_loss += tf.nn.l2_loss(var)

    l2_loss = tf.multiply(l2_regularization, tf.cast(l2_loss, tf.float64))

    total_loss = tf.cast(error, tf.float64) + l2_loss

    # create the adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate = learningRate).minimize(total_loss) # TODO: gaussian_noise_injection_std_dev=gaussianNoise

    # create the Dataset objects for the training input and label data
    training_inputs = tf.data.Dataset.from_generator(generator=lambda: list_of_trainig_inputs, output_types=tf.float32)
    training_labels = tf.data.Dataset.from_generator(generator=lambda: list_of_training_labels, output_types=tf.float32)

    # zip the training input and label datasets together
    training_dataset = tf.data.Dataset.zip((training_inputs, training_labels))

    # create the Dataset objects for the validation input and label data with padded values
    validation_inputs = tf.data.Dataset.from_generator(generator=lambda: list_of_validation_inputs, output_types=tf.float32)
    validation_labels = tf.data.Dataset.from_generator(generator=lambda: list_of_validation_labels, output_types=tf.float32)

    # zip the validation input and label data
    validation_input_label_dataset = tf.data.Dataset.zip((validation_inputs, validation_labels))

    # create Dataset objects for the validation levels, true values and true seasonality
    validation_levels = tf.data.Dataset.from_generator(generator=lambda: list_of_levels, output_types=tf.float32)
    validation_true_values = tf.data.Dataset.from_generator(generator=lambda: list_of_true_values, output_types=tf.float32)
    validation_true_seasonality = tf.data.Dataset.from_generator(generator=lambda: list_of_true_seasonality,
                                                            output_types=tf.float32)

    # zip the validation datasets except the inputs and labels
    validation_dataset = tf.data.Dataset.zip((validation_levels, validation_true_values, validation_true_seasonality))

    # define the expected shapes of data after padding
    padded_shapes = ([tf.Dimension(None), INPUT_SIZE], [tf.Dimension(None), OUTPUT_SIZE])

    INFO_FREQ = 1
    sMAPE_final_list = []

    # setup variable initialization
    init_op = tf.global_variables_initializer()

    with tf.Session() as session :
        session.run(init_op)

        for iscan in range(int(maxNumOfEpochs)):
            sMAPE_epoch_list = []
            print("Epoch->", iscan)

            # randomly shuffle the time series within the dataset
            training_dataset.shuffle(mbSize)

            for epochsize in range(int(maxEpochSize)):
                sMAPE_list = []

                # create the batches by padding the datasets to make the variable sequence lengths fixed within the individual batches
                padded_training_data_batches = training_dataset.padded_batch(batch_size = int(mbSize), padded_shapes = padded_shapes)

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

                if iscan % INFO_FREQ == 0:

                    # create a single batch from all the validation time series by padding the datasets to make the variable sequence lengths fixed
                    padded_validation_input_label_data = validation_input_label_dataset.padded_batch(batch_size = len(list_of_validation_inputs), padded_shapes = padded_shapes)

                    # create a single batch for the rest of the validation data without padding
                    validation_data_batches = validation_dataset.batch(batch_size = len(list_of_levels))

                    # get an iterator to the validation input, label data batch
                    validation_input_label_data_iterator = padded_validation_input_label_data.make_one_shot_iterator()

                    # get an iterator to the validation data batch
                    validation_data_batch_iterator = validation_data_batches.make_one_shot_iterator()

                    # access the validation input, label batch using the iterator
                    validation_input_label_data_batch = validation_input_label_data_iterator.get_next()

                    # access the validation data batch using the iterator
                    validation_data_batch = validation_data_batch_iterator.get_next()

                    # get the batch of validation inputs
                    validation_input_label_batch_value = session.run(validation_input_label_data_batch)

                    # get the next batch of the other validation data
                    validation_data_batch_value = session.run(validation_data_batch)

                    # get the output of the network for the validation input data batch
                    validation_output = session.run(dense_layer, feed_dict={input: validation_input_label_batch_value[0]})

                    # get the lengths of the output data
                    output_sequence_lengths = session.run(length(validation_output))

                    # iterate across the different time series
                    for i in range(validation_output.shape[0]) :

                        # convert the data to remove the preprocessing
                        true_seasonality_values = validation_data_batch_value[2][i]
                        level = validation_data_batch_value[1][i]
                        last_index = output_sequence_lengths[i] - 1
                        last_validation_output = validation_output[i, last_index]

                        converted_validation_output = true_seasonality_values + level + last_validation_output
                        actual_values = validation_input_label_batch_value[1][i, last_index]
                        converted_actual_values = actual_values + true_seasonality_values + level

                        # calculate the smape
                        sMAPE = np.mean(np.abs(converted_validation_output - converted_actual_values) /
                                        (np.abs(converted_validation_output) + np.abs(converted_actual_values))) * 2
                        sMAPE_list.append(sMAPE)

                sMAPEprint = np.mean(sMAPE_list)
                sMAPE_epoch_list.append(sMAPEprint)

            sMAPE_validation = np.mean(sMAPE_epoch_list)
            sMAPE_final_list.append(sMAPE_validation)
            print("smape: ", sMAPE_validation)

        sMAPE_final = np.mean(sMAPE_final_list)
        max_value = 1 / (sMAPE_final)

    return max_value

if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)

    init_points = 2
    num_iter = 30
    time_series = 15
    mini_batch_size = 10

    create_train_data()

    # using bayesian optimizer for hyperparameter optimization
    bayesian_optimization = BayesianOptimization(train_model, {'learningRate': (0.0001, 0.0008),
                                                     'lstmCellDimension': (10, 100),
                                                     'mbSize': (10, 30),
                                                     'maxEpochSize': (1, 1),
                                                     'maxNumOfEpochs': (3, 20),
                                                     'l2_regularization': (0.0001, 0.0008)
                                                     # 'gaussianNoise': (0.0001, 0.0008)
                                                        })

    bayesian_optimization.maximize(init_points = init_points, n_iter = num_iter)
    print(bayesian_optimization.res['max'])
    # train_model(learningRate=0.0013262220421187676, lstmCellDimension=2, mbSize=mbSize, maxEpochSize=1, maxNumOfEpochs=20,
    #             l2_regularization=0.00015753660121731034, gaussianNoise=0.00023780395225712772)
