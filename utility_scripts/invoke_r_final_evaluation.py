import subprocess

def invoke_r_script(args, moving_window, without_stl_decomposition):
    if moving_window:
        subprocess.call(["Rscript", "--vanilla", "error_calculator/moving_window/final_evaluation.R", args[0], args[1], args[2], args[3], args[4], args[5], args[6]])
    else:
        if without_stl_decomposition:
            subprocess.call(["Rscript", "--vanilla", "error_calculator/non_moving_window/without_stl_decomposition/final_evaluation.R", args[0], args[1], args[2], args[3], args[4], args[5]])
        else:
            subprocess.call(
                ["Rscript", "--vanilla", "error_calculator/non_moving_window/with_stl_decomposition/final_evaluation.R", args[0], args[1], args[2], args[3], args[4], args[5]])