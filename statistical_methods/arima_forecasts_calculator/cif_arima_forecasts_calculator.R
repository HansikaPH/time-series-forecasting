library(forecast)
# read the data
file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/cif-dataset-o6.txt"
cif_dataset_o6 <- readLines(file)
cif_dataset_o6 <- strsplit(cif_dataset_o6, ',')

file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/cif-dataset-o12.txt"
cif_dataset_o12 <- readLines(file)
cif_dataset_o12 <- strsplit(cif_dataset_o12, ',')

output_file_name_o6 = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/arima_forecasts/cif2016_O6.txt"
output_file_name_o12 = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/arima_forecasts/cif2016_O12.txt"

# clear the file content
close(file(output_file_name_o6, open = "w"))
close(file(output_file_name_o12, open = "w"))

# calculate the arima forecast
for (i in 1 : length(cif_dataset_o6)) {
    time_series = unlist(cif_dataset_o6[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

    fit = NULL
    if (length(time_series) < 24) {
        fit = auto.arima(ts(time_series, frequency = 12), seasonal = FALSE)
    }else {
        tryCatch({
            fit = auto.arima(ts(time_series, frequency = 12), lambda = 0)
        }, warning = function(e) {
          print(e)
        })
    }
    
    if(is.null(fit)){
      fit = auto.arima(ts(time_series, frequency = 12), stationary = TRUE)
    }
    
    forecasts = forecast(fit, h=6)$mean

    # write the arima forecasts to file
    write.table(t(forecasts), file=output_file_name_o6, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}

# calculate the arima forecast
for (i in 1 : length(cif_dataset_o12)) {
    time_series = unlist(cif_dataset_o12[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

    fit = NULL
    if (length(time_series) < 24) {
        fit = auto.arima(ts(time_series, frequency = 12), seasonal = FALSE)
    }else {
        tryCatch({
            fit = auto.arima(ts(time_series, frequency = 12), lambda = 0)
        }, warning = function(e) {
          print(e)
        })
    }
    
    if(is.null(fit)){
      fit = auto.arima(ts(time_series, frequency = 12), stationary = TRUE)
    }
    
    forecasts = forecast(fit, h=12)$mean

    # write the arima forecasts to file
    write.table(t(forecasts), file=output_file_name_o12, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}
