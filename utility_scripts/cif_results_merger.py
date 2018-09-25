import glob
import numpy as np

input_path = '../results/errors/'

output_file_mean_median = input_path + "mean_median_cif2016_"
output_file_ranked_error = input_path + "all_errors_cif2016_"

# get the list of all the files matching the regex
all_errors_files_o6 = [filename for filename in glob.iglob(input_path + "all_errors_cif2016_O6_" + "*")]
all_errors_files_o12 = [filename for filename in glob.iglob(input_path + "all_errors_cif2016_O12_" + "*")]

# read the files one by one and merge the content
for filename_o6, filename_o12 in zip(sorted(all_errors_files_o6), sorted(all_errors_files_o12)):
    filename_o6_object = open(filename_o6)
    filename_o12_object = open(filename_o12)

    filename_part = filename_o6.split(sep="all_errors_cif2016_O6_", maxsplit=1)[1]

    output_file_ranked_error_object = open(output_file_ranked_error + filename_part, "w")
    output_file_mean_median_object = open(output_file_mean_median + filename_part, "w")

    # read the errors from both files
    all_errors_o6 = [float(num) for num in filename_o6_object]
    all_errors_o12 = [float(num) for num in filename_o12_object]

    # merge the errors of the two files
    all_errors_array = np.array(all_errors_o12 + all_errors_o6, dtype=np.float32)

    # calculate the mean, median. std
    mean = np.mean(all_errors_array, axis=0)
    median = np.median(all_errors_array, axis=0)
    std = np.std(all_errors_array, axis=0)

    # write the mean, median, std to file
    output_file_mean_median_object.writelines(output_file_mean_median + filename_part)
    output_file_mean_median_object.writelines("mean_error:" + mean)
    output_file_mean_median_object.writelines("median_error:" + median)
    output_file_mean_median_object.writelines("std_error:" + std)

    # write all errors to file
    output_file_ranked_error_object.write(np.transpose(all_errors_array))

    # close all files
    filename_o6_object.close()
    filename_o12_object.close()
    output_file_mean_median_object.close()