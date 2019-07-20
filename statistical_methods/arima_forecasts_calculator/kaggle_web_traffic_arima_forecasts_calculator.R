library(forecast)
# read the data
kaggle_dataset <- read.csv(file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt", sep = ',', header = FALSE)
kaggle_dataset <- as.matrix(kaggle_dataset)

output_file_name = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/arima_forecasts/kaggle_web_traffic.txt"
original_forecasts = readLines(output_file_name)

indices = list(53, 54, 55, 56, 173, 174, 867)
# clear the file content
#close(file(output_file_name, open = "w"))

# calculate the arima forecasts
for (i in indices) {

    time_series = kaggle_dataset[i,]
    fit = NULL
   
    tryCatch({
        fit = auto.arima(ts(time_series, frequency = 7), lambda = 0)
    }, warning = function(e) {
      print(e)
    })
    
    if(is.null(fit)){
      fit = auto.arima(ts(time_series, frequency = 7), seasonal = FALSE)
    }
    forecasts = forecast(fit, h=59)$mean

    forecasts[forecasts<0] <- 0
    forecasts = round(forecasts)
    
    forecasts = as.data.frame(t(forecasts))
    string_forecasts = format_csv(forecasts, col_names = FALSE)
    string_forecasts = strsplit(string_forecasts, "\n")[[1]]
    
    original_forecasts[i] = string_forecasts

    # write the arima forecasts to file
    #write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}