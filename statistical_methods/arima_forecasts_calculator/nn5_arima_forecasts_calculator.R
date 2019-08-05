library(forecast)

# read the data
nn5_dataset <- read.csv(file = "./datasets/text_data/NN5/nn5_dataset.txt", sep = ',', header = FALSE)
nn5_dataset <- as.matrix(nn5_dataset)

output_file_name = "./results/arima_forecasts/nn5.txt"

unlink(output_file_name)

# calculate the arima forecasts
for (i in 1 : nrow(nn5_dataset)) {

    time_series = nn5_dataset[i,]
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
    forecasts = forecast(fit, h=56)$mean

    # write the arima forecasts to file
    write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}