import csv

data_file = "../../datasets/text_data/M3/M3C.csv"

train_data_file = "../../datasets/text_data/M3/Train_Dataset.csv"
results_file = "../../datasets/text_data/M3/Test_Dataset.csv"

with open(data_file, "r") as original_file, open(train_data_file, "w") as train_out, open(results_file, "w") as results_out:
    data_reader = csv.reader(original_file, delimiter=",")
    train_data_writer = csv.writer(train_out, delimiter=",")
    results_writer = csv.writer(results_out, delimiter=",")
    next(data_reader, None)
    for row in data_reader:
        number_of_columns = row[3]
        train_data = row[7 : 7 + int(number_of_columns)]
        train_data.insert(0, row[4])
        train_data.insert(0, row[0])

        results_data = row[7 + int(number_of_columns): len(row)]
        results_data.insert(0, row[4])
        results_data.insert(0, row[0])
        print(results_data)


        train_data_writer.writerow(train_data)
        results_writer.writerow(results_data)
