OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN3/moving_window/"
output_file_name = 'nn3_results.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

nn3_result_dataset <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN3/NN3_TEST_DATASET.csv",sep=';',header = FALSE)


# printing the results to the file
write.table(nn3_result_dataset, output_file_full_name, sep = ";", row.names = TRUE, col.names = FALSE)
