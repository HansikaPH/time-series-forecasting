# configs for the model training
class model_training_configs:
    VALIDATION_ERRORS_DIRECTORY = 'results/validation_errors/'
    INFO_FREQ = 1

# configs for the model testing
class model_testing_configs:
    RNN_FORECASTS_DIRECTORY = 'results/rnn_forecasts/'
    RNN_ERRORS_DIRECTORY = 'results/errors'
    PROCESSED_RNN_FORECASTS_DIRECTORY = '/results/processed_rnn_forecasts/'

# configs for hyperparameter tuning(SMAC3)
class hyperparameter_tuning_configs:
    SMAC_RUNCOUNT_LIMIT = 50

class gpu_configs:
    log_device_placement = False
