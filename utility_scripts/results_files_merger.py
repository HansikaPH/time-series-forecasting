import glob
import sys

dataset_name = sys.argv[1]
output_path = '../results/errors/'
output_file = output_path + dataset_name + "_all_results"

output_file_object = open(output_file, "w")

# get the list of all the files matching the regex except the destination file
files = [filename for filename in glob.iglob(output_path + dataset_name + "_*") if not filename == output_file]

# read the files one by one and merge the content
for filename in sorted(files):
    with open(filename) as file_object:
        # read content from results file
        content = file_object.readlines()

        # write content to final output file
        output_file_object.writelines(filename)
        output_file_object.writelines(content)
        output_file_object.writelines("\n\n")
        file_object.close()

output_file_object.close()