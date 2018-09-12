# configs for the model training
class model_training_configs:
    INFO_FREQ = 1

# configs for the model testing
class model_testing_configs:
    FORECASTS_DIRECTORY = 'results/forecasts/'

# configs for hyperparameter tuning(bayesian optimization/SMAC3)
class hyperparameter_tuning_configs:
    BAYESIAN_INIT_POINTS = 2
    BAYESIAN_NUM_ITER = 30
    SMAC_RUNCOUNT_LIMIT = 1

class training_data_configs:
    SHUFFLE_BUFFER_SIZE = 10000