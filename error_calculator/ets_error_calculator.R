library(TSrepr)

args <- commandArgs(trailingOnly = TRUE)
forecast_file = args[1]
actual_forecasts_file = args[2]
snaive_forecasts_file = args[3]
dataset_name = args[4]

base_dir = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/"
errors_dir = paste(base_dir, "results/errors/", sep="")
forecast_file = paste(base_dir, forecast_file, sep="")
actual_forecasts_file = paste(base_dir, actual_forecasts_file, sep="")
snaive_forecasts_file = paste(base_dir, snaive_forecasts_file, sep="")

errors_file_full_name_mean_median = paste(errors_dir, "mean_median_", dataset_name, "_ets.txt", sep="")
SMAPE_file_full_name_all_errors = paste(errors_dir, "all_smape_errors_", dataset_name, "_ets.txt", sep="")
MASE_file_full_name_all_errors = paste(errors_dir, "all_mase_errors_", dataset_name, "_ets.txt", sep="")

# read the forecasts
forecasts_df=read.csv(forecast_file, header = F, sep = ",")

# read the actual forecasts
actual_forecasts_df=read.csv(file=actual_forecasts_file,sep=';',header = FALSE)
actual_forecasts_df = actual_forecasts_df[,-1]

# read the snaive forecasts
snaive_forecasts_df = read.csv(snaive_forecasts_file, header = F, sep=",")

# calculating the SMAPE
time_series_wise_SMAPE <- 2*abs(forecasts_df-actual_forecasts_df)/(abs(forecasts_df)+abs(actual_forecasts_df))
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
mase_vector = NULL

for(k in 1 :nrow(forecasts_df)){
  mase_vector[k] = mase(unlist(actual_forecasts_df[k,]), unlist(forecasts_df[k,]), unlist(snaive_forecasts_df[k,]))
}

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