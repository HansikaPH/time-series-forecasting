import argparse
import csv

argument_parser = argparse.ArgumentParser("Calculate the time series length")
argument_parser.add_argument('--data_file', required=True, help='The full name of the data file')
argument_parser.add_argument('--output_file', required=True, help='The full name of the output file')

args = argument_parser.parse_args()
data_file = args.data_file
output_file = args.output_file

lengths_list =[]

with open(data_file) as input:
    for line in input:
        length = line.count(",") + 1
        lengths_list.append(length)

with open(output_file, "w") as output:
    for item in lengths_list:
        output.write("%s\n" % item)