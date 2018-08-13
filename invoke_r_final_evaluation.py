import subprocess

def invoke_r_script(args, moving_window):
    if moving_window:
        subprocess.call(["Rscript", "--vanilla", "error_calculator/moving_window/final_evaluation.R", args[0], args[1], args[2], args[3], args[4], args[5], args[6]])
    else:
        subprocess.call(["Rscript", "--vanilla", "error_calculator/non_moving_window/final_evaluation.R", args[0], args[1], args[2], args[3], args[4], args[5]])