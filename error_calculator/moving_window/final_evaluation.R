args <- commandArgs(trailingOnly = TRUE)
forecast_file_path = args[1]
errors_file_name = args[2]
txt_test_file_name = args[3]
actual_results_file_name = args[4]
input_size = as.numeric(args[5])
output_size = as.numeric(args[6])
contain_zero_values = as.numeric(args[7])

root_directory = paste(dirname(getwd()), "time-series-forecasting", sep="/")

# errors file name
errors_directory = paste(root_directory, "results/errors", sep="/")
errors_file_full_name = paste(errors_directory, errors_file_name, sep='/')

# actual results file name
actual_results_file_full_name = paste(root_directory, actual_results_file_name, sep="/")
actual_results=read.csv(file=actual_results_file_full_name,sep=';',header = FALSE)

# text test data file name
txt_test_file_full_name = paste(root_directory, txt_test_file_name, sep="/")
txt_test_df=read.csv(file=txt_test_file_name,sep = " ",header = FALSE)

# forecasts file name
forecasts_file_full_name = paste(root_directory, forecast_file_path, sep="/")
forecasts_df=read.csv(forecasts_file_full_name, header = F, sep = ",")

names(actual_results)[1]="Series"

actual_results <- actual_results[,-1]

# take the transpose of the dataframe
value <- t(txt_test_df[1])

indexes <- length(value) - match(unique(value), rev(value)) + 1

uniqueindexes <- unique(indexes)

actual_results_df <- actual_results[rowSums(is.na(actual_results)) == 0,]

converted_forecasts_df = NULL
converted_forecasts_matrix = matrix(nrow = nrow(forecasts_df), ncol = output_size)

for(k in 1 :nrow(forecasts_df)){
  one_ts_forecasts = as.numeric(forecasts_df[k,])
  finalindex <- uniqueindexes[k]
  one_line_test_data = as.numeric(txt_test_df[finalindex,])
  
  level_value = one_line_test_data[input_size + 3]
  seasonal_values = one_line_test_data[input_size + 4:length(one_line_test_data)]
  
  for (ii in 1:output_size) {
    converted_value = exp(one_ts_forecasts[ii] + level_value + seasonal_values[ii])
    if(contain_zero_values == 1){
      converted_value = converted_value -1
    }
    converted_forecasts_df[ii] =  converted_value 
  }
  converted_forecasts_matrix[k,] = converted_forecasts_df
}

sMAPEPerSeries1 <- rowMeans(2*abs(converted_forecasts_matrix-actual_results_df)/(abs(converted_forecasts_matrix)+abs(actual_results_df)), na.rm=TRUE)

mean_error = mean(sMAPEPerSeries1)
std_error = sd(sMAPEPerSeries1)
print(mean_error)
print(std_error)

write.table(c(mean_error, std_error), errors_file_full_name)