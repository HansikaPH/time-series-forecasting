from bayes_opt import BayesianOptimization
import tensorflow as tf
import sys

sys.path.insert(0, '../')
from stacking_model_trainer import StackingModelTrainer

LSTM_USE_PEEPHOLES = True
LSTM_USE_STABILIZATION = True
BIAS = False

# Input/Output Window sizes
INPUT_SIZE = 15
OUTPUT_SIZE = 12

# Training and Validation file paths.
binary_train_file_path = '../../DataSets/CIF 2016/binary_files/stl_12i15.tfrecords'
binary_validation_file_path = '../../DataSets/CIF 2016/binary_files/stl_12i15v.tfrecords'

learning_rate = 0.0

# function to create the optimizer
def optimizer_fn(total_loss):
    return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)

# Training the time series
def train_model(rate_of_learning, no_hidden_layers, lstm_cell_dimension, minibatch_size, max_epoch_size, max_num_of_epochs, l2_regularization, gaussian_noise_stdev):

    global learning_rate
    learning_rate = rate_of_learning

    stacking_model_trainer = StackingModelTrainer(use_bias = BIAS,
                                                use_peepholes = LSTM_USE_PEEPHOLES,
                                                input_size = INPUT_SIZE,
                                                output_size = OUTPUT_SIZE,
                                                binary_train_file_path = binary_train_file_path,
                                                binary_validation_file_path = binary_validation_file_path)

    error = stacking_model_trainer.train_model(learning_rate = learning_rate,
                        no_hidden_layers = no_hidden_layers,
                        lstm_cell_dimension = lstm_cell_dimension,
                        minibatch_size = minibatch_size,
                        max_epoch_size = max_epoch_size,
                        max_num_of_epochs = max_num_of_epochs,
                        l2_regularization = l2_regularization,
                        gaussian_noise_stdev = gaussian_noise_stdev,
                        optimizer_fn = optimizer_fn)
    return -1 * error

if __name__ == '__main__':

    init_points = 2
    num_iter = 30

    # using bayesian optimizer for hyperparameter optimization
    bayesian_optimization = BayesianOptimization(train_model, {'rate_of_learning': (0.0001, 0.0008),
                                                               'no_hidden_layers': (1, 5),
                                                                'lstm_cell_dimension': (50, 100),
                                                                'minibatch_size': (10, 30),
                                                                'max_epoch_size': (1, 3),
                                                                'max_num_of_epochs': (3, 20),
                                                                      'l2_regularization': (0.0001, 0.0008),
                                                                      'gaussian_noise_stdev': (0.0001, 0.0008)
                                                                      })

    bayesian_optimization.maximize(init_points = init_points, n_iter = num_iter)

