library(forecast)
# read the data
kaggle_dataset <- read.csv(file = "./datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt", sep = ',', header = FALSE)
kaggle_dataset <- as.matrix(kaggle_dataset)

output_file_name = "./results/arima_forecasts/kaggle_web_traffic.txt"

unlink(output_file_name)

# calculate the arima forecasts
for (i in 1 : nrow(kaggle_dataset)) {

    time_series = kaggle_dataset[i,]
    fit = NULL
    forecasts = NULL
   
    tryCatch({
        fit = auto.arima(ts(time_series, frequency = 7), lambda = 0)
    }, warning = function(e) {
      print(e)
    })
    
    if(is.null(fit)){
      tryCatch({
        fit = auto.arima(ts(time_series, frequency = 7))
      }, warning = function(e) {
        print(e)
      })
      if(is.null(fit)){
        fit = auto.arima(ts(time_series, frequency = 7), seasonal = FALSE)
      }
    }
    forecasts = forecast(fit, h=59)$mean

    forecasts[forecasts<0] <- 0
    forecasts = round(forecasts)

    # write the arima forecasts to file
    write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}