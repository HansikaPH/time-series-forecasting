library(forecast)

# read the data
file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M3/Train_Dataset.csv"
m3_dataset <- readLines(file)
m3_dataset <- strsplit(m3_dataset, ',')

output_file_name = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/arima_forecasts/m3.txt"

# clear the file content
close(file(output_file_name, open = "w"))

# calculate the arima forecast
for (i in 1 : length(m3_dataset)) {
    time_series = unlist(m3_dataset[i], use.names = FALSE)
    time_series = as.numeric(time_series[2 : length(time_series)])
    fit = NULL
    
    tryCatch({
        fit = auto.arima(ts(time_series, frequency = 12), lambda = 0)
    }, warning = function(e) {
        print(e)
    })
    
    if(is.null(fit)){
      fit = auto.arima(ts(time_series, frequency = 12), seasonal = FALSE)
    }
    forecasts = forecast(fit, h=18)$mean

    # write the arima forecasts to file
    write.table(t(forecasts), file=output_file_name, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}

