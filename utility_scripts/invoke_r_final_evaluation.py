import subprocess
from configs.global_configs import model_testing_configs

def invoke_r_script(args, moving_window):
    if moving_window:
        subprocess.call(["Rscript", "--vanilla", "error_calculator/moving_window/final_evaluation.R", args[0], model_testing_configs.RNN_ERRORS_DIRECTORY, model_testing_configs.PROCESSED_RNN_FORECASTS_DIRECTORY, args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]])
    else:
        subprocess.call(["Rscript", "--vanilla", "error_calculator/non_moving_window/final_evaluation.R", args[0], model_testing_configs.RNN_ERRORS_DIRECTORY, model_testing_configs.PROCESSED_RNN_FORECASTS_DIRECTORY, args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]])
