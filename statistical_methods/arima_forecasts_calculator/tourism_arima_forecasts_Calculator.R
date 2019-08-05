library(forecast)

# read the data
file = "./datasets/text_data/Tourism/tourism_data.csv"
tourism_dataset <- readLines(file)
tourism_dataset <- strsplit(tourism_dataset, ',')

output_file_name = "./results/arima_forecasts/tourism.txt"

# delete the file if existing
unlink(output_file_name)

# calculate the arima forecasts
for (i in 1 : length(tourism_dataset)) {
    time_series = unlist(tourism_dataset[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

    fit = NULL
    forecasts = NULL
  
    tryCatch({
        fit = auto.arima(ts(time_series, frequency = 12), lambda = 0)
    }, warning = function(e) {
      print(e)
    })
    
    if(is.null(fit)){
      tryCatch({
        fit = auto.arima(ts(time_series, frequency = 12))
      }, warning = function(e) {
        print(e)
      })
      if(is.null(fit)){
        fit = auto.arima(ts(time_series, frequency = 12), seasonal = FALSE)
      }
    }

    forecasts = forecast(fit, h = 24)$mean

    # write the arima forecasts to file
    write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}