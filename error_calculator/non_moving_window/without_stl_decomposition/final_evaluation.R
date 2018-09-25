library(ggplot2)
library(reshape2)

args <- commandArgs(trailingOnly = TRUE)
forecast_file_path = args[1]
errors_file_name = args[2]
txt_test_file_name = args[3]
actual_results_file_name = args[4]
output_size = as.numeric(args[5])
contain_zero_values = as.numeric(args[6])

root_directory = paste(dirname(getwd()), "time-series-forecasting", sep="/")

# errors file name
errors_directory = paste(root_directory, "results/errors", sep="/")
errors_file_name_mean_median = paste("mean_median", errors_file_name, sep='_')
errors_file_name_all_errors = paste("all_errors", errors_file_name, sep='_')
errors_file_full_name_mean_median = paste(errors_directory, errors_file_name_mean_median, sep='/')
errors_file_full_name_all_errors = paste(errors_directory, errors_file_name_all_errors, sep='/')

# actual results file name
actual_results_file_full_name = paste(root_directory, actual_results_file_name, sep="/")
actual_results=read.csv(file=actual_results_file_full_name,sep=';',header = FALSE)

# text test data file name
txt_test_file_full_name = paste(root_directory, txt_test_file_name, sep="/")
txt_test_df = readLines(txt_test_file_full_name)
txt_test_df = strsplit(txt_test_df, " ")

# forecasts file name
forecasts_file_full_name = paste(root_directory, forecast_file_path, sep="/")
forecasts_df=read.csv(forecasts_file_full_name, header = F, sep = ",")

names(actual_results)[1]="Series"

actual_results <- actual_results[,-1]
actual_results_df <- actual_results[rowSums(is.na(actual_results)) == 0,]

converted_forecasts_df = NULL
converted_forecasts_matrix = matrix(nrow = nrow(forecasts_df), ncol = output_size)

for(k in 1 :nrow(forecasts_df)){
  one_ts_forecasts = as.numeric(forecasts_df[k,])
  one_line_test_data = as.numeric(unlist(txt_test_df[k]))

  level_value = one_line_test_data[length(one_line_test_data) - output_size]
  # seasonal_values = tail(one_line_test_data, output_size)

  for (ii in 1:output_size) {
    converted_value = exp(one_ts_forecasts[ii] + level_value
# + seasonal_values[ii]
)
    if(contain_zero_values == 1){
      converted_value = converted_value -1
    }
    converted_forecasts_df[ii] =  converted_value
  }
  converted_forecasts_matrix[k,] = converted_forecasts_df
}

sMAPEPerSeries1 <- rowMeans(2*abs(converted_forecasts_matrix-actual_results_df)/(abs(converted_forecasts_matrix)+abs(actual_results_df)), na.rm=TRUE)

mean_error = mean(sMAPEPerSeries1)
median_error = median(sMAPEPerSeries1)
std_error = sd(sMAPEPerSeries1)

mean_error = paste("mean_error", mean_error, sep=":")
median_error = paste("median_error", median_error, sep=":")
std_error = paste("std_error", std_error, sep=":")
print(mean_error)
print(median_error)
print(std_error)

file_connection = file(errors_file_full_name_mean_median)
writeLines(c(mean_error, median_error, std_error, "\n"), file_connection)
close(file_connection)
write.table(sMAPEPerSeries1, errors_file_full_name_all_errors, row.names=FALSE, col.names=FALSE)

# errors_data_frame = as.data.frame(sMAPEPerSeries1)
# names(errors_data_frame) = "Errors"
# errors_data_frame$time_series_no = (1: nrow(errors_data_frame))
# print(errors_data_frame[errors_data_frame$Errors >= 0.6, ])
#
# # plot the errors
# ggplot(data = errors_data_frame, aes(x =time_series_no, y=Errors)) + geom_line(color = "#00AFBC", size = 1)
#
#
# print(sMAPEPerSeries1)
#
# print(median(sMAPEPerSeries1))
# print(mean(errors_data_frame$Errors[errors_data_frame$Errors < 0.6]))
#
# # prepare the actual results for plotting
# transposed_actual_results <- as.data.frame(t(actual_results[,-1]))
# names(transposed_actual_results)[1:ncol(transposed_actual_results)]=paste('ts',(1:(ncol(transposed_actual_results))),sep='_')
# transposed_actual_results$actual_result_no = (1: nrow(transposed_actual_results))
#
# # prepare the forecasts for plotting
# transposed_forecasts <- as.data.frame(t(converted_forecasts_matrix))
# names(transposed_forecasts)[1:ncol(transposed_forecasts)]=paste('ts',(1:(ncol(transposed_forecasts))),sep='_')
# transposed_forecasts$forecast_no = (1: nrow(transposed_forecasts))
#
# # plot the actual results and forecasts
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_2)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_2))+geom_line(color = "#00AFBB", size = 1)
#
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_55)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_55))+geom_line(color = "#00AFBB", size = 1)
#
# # ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_17)) + geom_line(color = "#00AFBB", size = 1)
# # ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_17))+geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_21)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_21))+geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_28)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_28))+geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_48)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_48))+geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_49)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_49))+geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_65)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_65))+geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_72)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_72))+geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_actual_results, aes(x =actual_result_no, y=ts_108)) + geom_line(color = "#00AFBB", size = 1)
# ggplot(data = transposed_forecasts, aes(x =forecast_no, y=ts_108))+geom_line(color = "#00AFBB", size = 1)