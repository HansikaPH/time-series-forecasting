OUTPUT_DIR="./datasets/text_data/M3/"

file <-read.csv(file="./datasets/text_data/M3/Test_Dataset.csv",sep=';',header = FALSE)
m3_result_dataset <-as.data.frame(file[,-1])

# printing the results to the file

write.table(m3_result_dataset[1:474,], paste(OUTPUT_DIR, "m3_result_monthly_micro.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m3_result_dataset[475:808,], paste(OUTPUT_DIR, "m3_result_monthly_industry.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m3_result_dataset[809:1120,], paste(OUTPUT_DIR, "m3_result_monthly_macro.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m3_result_dataset[1121:1265,], paste(OUTPUT_DIR, "m3_result_monthly_finance.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m3_result_dataset[1266:1376,], paste(OUTPUT_DIR, "m3_result_monthly_demo.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m3_result_dataset[1377:nrow(m3_result_dataset),], paste(OUTPUT_DIR, "m3_result_monthly_other.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)