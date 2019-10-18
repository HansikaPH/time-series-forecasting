# configs for the model training
# class model_training_configs:
#     VALIDATION_ERRORS_DIRECTORY = 'results/validation_errors/'
#     OPTIMIZED_CONFIG_DIRECTORY = 'results/optimized_configurations/'
#     INFO_FREQ = 1

# configs for the model testing
class model_testing_configs:
    RNN_FORECASTS_DIRECTORY = 'results/rnn_forecasts/'
    RNN_ERRORS_DIRECTORY = 'results/errors'
    PROCESSED_RNN_FORECASTS_DIRECTORY = '/results/processed_rnn_forecasts/'

# configs for hyperparameter tuning(SMAC3)
class hyperparameter_tuning_configs:
    SMAC_RUNCOUNT_LIMIT = 50
    OPTIMIZED_CONFIG_DIRECTORY = 'results/optimized_configurations/'

class training_data_configs:
    SHUFFLE_BUFFER_SIZE = 1000

class gpu_configs:
    log_device_placement = False