OUTPUT_DIR="./datasets/text_data/M4/"

file <-read.csv(file="./datasets/text_data/M4/Monthly-test.csv",sep=';',header = FALSE)
m4_result_dataset <-as.data.frame(file[,-1])

# printing the results to the file

# macro data
write.table(m4_result_dataset[1:10016,], paste(OUTPUT_DIR, "m4_result_monthly_macro.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m4_result_dataset[10017:20991,], paste(OUTPUT_DIR, "m4_result_monthly_micro.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m4_result_dataset[20992:26719,], paste(OUTPUT_DIR, "m4_result_monthly_demo.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m4_result_dataset[26720:36736,], paste(OUTPUT_DIR, "m4_result_monthly_industry.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m4_result_dataset[36737:47723,], paste(OUTPUT_DIR, "m4_result_monthly_finance.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)

# macro data
write.table(m4_result_dataset[47724:nrow(m4_result_dataset),], paste(OUTPUT_DIR, "m4_result_monthly_other.txt", sep = ''), sep = ";", row.names = TRUE, col.names = FALSE)