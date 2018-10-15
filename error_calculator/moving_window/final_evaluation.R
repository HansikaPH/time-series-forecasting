library(ggplot2)
library(reshape2)
library(TSrepr)

args <- commandArgs(trailingOnly = TRUE)
rnn_forecast_file_path = args[1]
snaive_forecasts_file_path = args[2]
errors_file_name = args[3]
txt_test_file_name = args[4]
actual_results_file_name = args[5]
input_size = as.numeric(args[6])
output_size = as.numeric(args[7])
contain_zero_values = as.numeric(args[8])

root_directory = paste(dirname(getwd()), "time-series-forecasting", sep="/")

# errors file name
errors_directory = paste(root_directory, "results/errors", sep="/")
errors_file_name_mean_median = paste("mean_median", errors_file_name, sep='_')
SMAPE_file_name_all_errors = paste("all_smape_errors", errors_file_name, sep='_')
MASE_file_name_all_errors = paste("all_mase_errors", errors_file_name, sep='_')
errors_file_full_name_mean_median = paste(errors_directory, errors_file_name_mean_median, sep='/')
SMAPE_file_full_name_all_errors = paste(errors_directory, SMAPE_file_name_all_errors, sep='/')
MASE_file_full_name_all_errors = paste(errors_directory, MASE_file_name_all_errors, sep='/')

# actual results file name
actual_results_file_full_name = paste(root_directory, actual_results_file_name, sep="/")
actual_results=read.csv(file=actual_results_file_full_name,sep=';',header = FALSE)

# text test data file name
txt_test_file_full_name = paste(root_directory, txt_test_file_name, sep="/")
txt_test_df=read.csv(file=txt_test_file_full_name,sep = " ",header = FALSE)

# rnn_forecasts file name
forecasts_file_full_name = paste(root_directory, rnn_forecast_file_path, sep="/")
forecasts_df=read.csv(forecasts_file_full_name, header = F, sep = ",")

# reading in the snaive forecasts file
snaive_forecasts_df = read.csv(snaive_forecasts_file_path, header = F, sep=",")

names(actual_results)[1]="Series"
actual_results <- actual_results[,-1]

# take the transpose of the dataframe
value <- t(txt_test_df[1])

indexes <- length(value) - match(unique(value), rev(value)) + 1

uniqueindexes <- unique(indexes)

actual_results_df <- actual_results[rowSums(is.na(actual_results)) == 0,]

converted_forecasts_df = NULL
converted_forecasts_matrix = matrix(nrow = nrow(forecasts_df), ncol = output_size)

mase_vector = NULL

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
  mase_vector[k] = mase(unlist(actual_results_df[k,]), converted_forecasts_df, unlist(snaive_forecasts_df[k,]))
}

# calculating the SMAPE
time_series_wise_SMAPE <- 2*abs(converted_forecasts_matrix-actual_results_df)/(abs(converted_forecasts_matrix)+abs(actual_results_df))
SMAPEPerSeries <- rowMeans(time_series_wise_SMAPE, na.rm=TRUE)

mean_SMAPE = mean(SMAPEPerSeries)
median_SMAPE = median(SMAPEPerSeries)
std_SMAPE = sd(SMAPEPerSeries)

mean_SMAPE = paste("mean_SMAPE", mean_SMAPE, sep=":")
median_SMAPE = paste("median_SMAPE", median_SMAPE, sep=":")
std_SMAPE = paste("std_SMAPE", std_SMAPE, sep=":")
print(mean_SMAPE)
print(median_SMAPE)
print(std_SMAPE)

# MASE
mean_MASE = mean(mase_vector)
median_MASE = median(mase_vector)
std_MASE = sd(mase_vector)

mean_MASE = paste("mean_MASE", mean_MASE, sep=":")
median_MASE = paste("median_MASE", median_MASE, sep=":")
std_MASE = paste("std_MASE", std_MASE, sep=":")
print(mean_MASE)
print(median_MASE)
print(std_MASE)

# writing the SMAPE results to file
write(c(mean_SMAPE, median_SMAPE, std_SMAPE, "\n"), file=errors_file_full_name_mean_median, append=FALSE)
write.table(SMAPEPerSeries, SMAPE_file_full_name_all_errors, row.names=FALSE, col.names=FALSE)

# writing the MASE results to file
write(c(mean_MASE, median_MASE, std_MASE, "\n"), file=errors_file_full_name_mean_median, append=TRUE)
write.table(mase_vector, MASE_file_full_name_all_errors, row.names=FALSE, col.names=FALSE)