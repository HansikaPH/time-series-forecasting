import subprocess
from configs.global_configs import model_testing_configs

def invoke_script(args):
    subprocess.call(["Rscript", "--vanilla", model_testing_configs.TESTING_ERROR_CALCULATOR_DIRECTORY] + args)