OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/train_data.csv",sep=',',header = TRUE)
kaggle_dataset <-as.data.frame(file[,-1])

output_file_name = 'kaggle_web_traffic_dataset.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

kaggle_dataset[is.na(kaggle_dataset)] = 0

# printing the dataset to the file
write.table(kaggle_dataset, output_file_full_name, sep = ",", row.names = FALSE, col.names = FALSE)
