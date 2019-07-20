import csv

file = "../../datasets/text_data/Electricity/non_solar_data.csv"
train_file = "../../datasets/text_data/Electricity/original_electricity_data.csv"
results_file = "../../datasets/text_data/Electricity/electricity_results.csv"

train_file_object = open(train_file, "a")
train_file_writer = csv.writer(train_file_object)
results_file_object = open(results_file, "a")
results_file_Writer = csv.writer(results_file_object, delimiter=";")
i = 0

# create the train test files
with open(file) as file_object:
    for line in file_object:
        i = i + 1
        splits = line.strip().split(sep=",")
        test = splits[-12:]
        train = splits[0:len(splits) - 12]
        test = ["ts" + str(i)] + test
        train_file_writer.writerow(train)
        results_file_Writer.writerow(test)