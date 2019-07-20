library(forecast)
library(tidyverse)

# read the data
nn5_dataset <- read.csv(file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/nn5_dataset.txt", sep = ',', header = FALSE)
nn5_dataset <- as.matrix(nn5_dataset)

output_file_name = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/arima_forecasts/nn5.txt"
original_forecasts = readLines(output_file_name)

indices = list(1,2,3,4,5,7,8,11,13,14,15,16,18,19,20,22,23,25,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,62,63,64,65,66,67,68,70,71,72,74,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,95,96,97,98,99,100,101,102,105,106,107,108,109,110,111)
# clear the file content
#close(file(output_file_name, open = "w"))

# calculate the arima forecasts
for (i in indices) {

    time_series = nn5_dataset[i,]
    fit = NULL
     
    tryCatch({
        fit = auto.arima(ts(time_series, frequency = 7), lambda = 0)
    }, warning = function(e) {
      print(e)
    })
    
    if(is.null(fit)){
      fit = auto.arima(ts(time_series, frequency = 7), seasonal = FALSE)
    }
    forecasts = forecast(fit, h=56)$mean
    forecasts = as.data.frame(t(forecasts))
    string_forecasts = format_csv(forecasts, col_names = FALSE)
    string_forecasts = strsplit(string_forecasts, "\n")[[1]]
    
    original_forecasts[i] = string_forecasts

    # write the arima forecasts to file
    #write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}