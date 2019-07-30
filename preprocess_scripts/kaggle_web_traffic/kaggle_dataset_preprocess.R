OUTPUT_DIR="./datasets/text_data/kaggle_web_traffic/"

file <-read.csv(file="./datasets/text_data/kaggle_web_traffic/train_data.csv",sep=',',header = TRUE)
kaggle_dataset <-as.data.frame(file[,-1])

output_file_name = 'kaggle_web_traffic_dataset.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

kaggle_dataset[is.na(kaggle_dataset)] = 0

# printing the dataset to the file
write.table(kaggle_dataset, output_file_full_name, sep = ",", row.names = FALSE, col.names = FALSE)
