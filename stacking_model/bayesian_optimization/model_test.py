import sys
import csv
import tensorflow as tf

sys.path.insert(0, '../')
from stacking_model_tester import StackingModelTester

# Input/Output Window size.
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# LSTM specific configurations.
LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

binary_train_file_path = '../../DataSets/CIF 2016/binary_files/stl_12i15v.tfrecords'
binary_test_file_path = '../../DataSets/CIF 2016/binary_files/cif12test.tfrecords'

# function to create the optimizer
def optimizer_fn(total_loss):
    return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)

if __name__ == '__main__':

    # optimized hyperparameters
    no_hidden_layers = 5
    max_no_of_epochs = 20
    max_epoch_size = 3
    learning_rate = 0.0006229434571986981
    lstm_cell_dimension = 80
    l2_regularization = 0.0005950321895642069
    minibatch_size = 10.367370648328363
    gaussian_noise_std = 0.0004274864418226984

    generic_model_tester = StackingModelTester(use_bias = BIAS,
                                                use_peepholes = LSTM_USE_PEEPHOLES,
                                                input_size = INPUT_SIZE,
                                                output_size = OUTPUT_SIZE,
                                                binary_train_file_path = binary_train_file_path,
                                                binary_test_file_path = binary_test_file_path)

    list_of_forecasts = generic_model_tester.test_model(no_hidden_layers = no_hidden_layers,
                        lstm_cell_dimension = lstm_cell_dimension,
                        minibatch_size = minibatch_size,
                        max_epoch_size = max_epoch_size,
                        max_num_of_epochs = max_no_of_epochs,
                        l2_regularization = l2_regularization,
                        gaussian_noise_stdev = gaussian_noise_std,
                        optimizer_fn = optimizer_fn)

    forecast_file_path = sys.argv[1]

    with open(forecast_file_path, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(list_of_forecasts)
