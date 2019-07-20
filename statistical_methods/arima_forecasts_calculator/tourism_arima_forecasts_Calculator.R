library(forecast)
library(tidyverse)

# read the data
file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Tourism/tourism_data.csv"
tourism_dataset <- readLines(file)
tourism_dataset <- strsplit(tourism_dataset, ',')

output_file_name = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/arima_forecasts/tourism.txt"
original_forecasts = readLines(output_file_name)

indices = list(45,61,63,64,82,97,109,111,116,117,133,134,142,145,160,163,165,167,168,170,171,175,184,203,204,218,219,220,221,227,228,229,230,232,233,234,235,244,249,250,253,254,255,258,260,261,262,263,267,287,290,291,307,308,311,315,316,317,318,332,333,343)
# clear the file content
#close(file(output_file_name, open = "w"))

# calculate the arima forecasts
for (i in indices) {
    time_series = unlist(tourism_dataset[i], use.names = FALSE)
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

    forecasts = forecast(fit, h = 24)$mean
    forecasts = as.data.frame(t(forecasts))
    string_forecasts = format_csv(forecasts, col_names = FALSE)
    string_forecasts = strsplit(string_forecasts, "\n")[[1]]
    
    original_forecasts[i] = string_forecasts

    # write the arima forecasts to file
    #write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}