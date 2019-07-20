library(forecast)
library(tidyverse)

# read the data
file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M4/original_m4_dataset.csv"
m4_dataset <- readLines(file)
m4_dataset <- strsplit(m4_dataset, ',')

output_file_name = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/arima_forecasts/m4.txt"
original_forecasts = readLines(output_file_name)

# clear the file content
#close(file(output_file_name, open = "w"))

indices = list(562,927,949,959,1485,2362,2364,2681,3162,3166,6201,7055,7058,7159,7267,7314,8307,8308,8647,8958,9894,10200,11261,11333,11354,13745,14728,14819,16734,16965,17306,17312,17325,17802,18807,18817,19021,19259,21057,21352,27193,27619,27824,28101,28280,28503,28627,28628,28629,28639,28642,29244,29256,30508,32682,33462,34506,35197,37657,37661,39179,41197,41199,41993,41995)

# calculate the arima forecast
for (i in indices) {
    time_series = unlist(m4_dataset[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

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
    forecasts = as.data.frame(t(forecasts))
    string_forecasts = format_csv(forecasts, col_names = FALSE)
    string_forecasts = strsplit(string_forecasts, "\n")[[1]]
    
    original_forecasts[i] = string_forecasts
    
    # write the arima forecasts to file
    #write.table(t(forecasts), file=output_file_name, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}

