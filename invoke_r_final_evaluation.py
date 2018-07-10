import subprocess

def invoke_r_script(args):
    subprocess.call(["Rscript", "--vanilla", "final_evaluation.R", args[0], args[1], args[2], args[3], args[4], args[5], args[6]])