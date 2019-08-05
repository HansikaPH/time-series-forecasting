OUTPUT_DIR="./datasets/text_data/kaggle_web_traffic/"
output_file_name = 'kaggle_web_traffic_results.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

file <-read.csv(file="./datasets/text_data/kaggle_web_traffic/test_data.csv",sep=',',header = TRUE)
kaggle_result_dataset <-as.data.frame(file[,-1])

kaggle_result_dataset[is.na(kaggle_result_dataset)] = 0

print(head(kaggle_result_dataset))

# printing the results to the file
write.table(kaggle_result_dataset, output_file_full_name, sep = ";", row.names = TRUE, col.names = FALSE)
