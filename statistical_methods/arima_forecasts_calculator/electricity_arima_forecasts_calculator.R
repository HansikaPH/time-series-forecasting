library(forecast)
library(tidyverse)

# read the data
file = "./datasets/text_data/Electricity/original_electricity_data.csv"
electricity_dataset <- readLines(file)
electricity_dataset <- strsplit(electricity_dataset, ',')

output_file_name = "./results/arima_forecasts/electricity.txt"
original_forecasts = readLines(output_file_name)

indices = list(32,34,35,62,112,137,139,166,167,210,272,333,381,395,404,413,445,463,494,506,516,547,557,576,602,626,641,659,684,701,768,780,796,815,830,845,866,885,888,890,949,969,974,989,1059,1096,1109,1139,1164,1170,1216,1236,1268,1304,1314,1328,1336,1343,1356,1358,1389,1457,1470,1508,1512,1535,1569,1575,1641,1643,1663,1740,1840,1860,1902,1958,1970,1981,1995,2000,2024,2051,2069,2101,2181,2216,2254,2276,2320,2349,2351,2374,2378,2407,2415,2521,2533,2565,2610,2692,2700,2706,2906,2950,3119,3168,3189,3280,3300,3361,3394,3453, 3454)
# clear the file content
#close(file(output_file_name, open = "w"))

# calculate the arima forecasts
for (i in indices) {
    time_series = unlist(electricity_dataset[i], use.names = FALSE)
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

    forecasts = forecast(fit, h=12)$mean
    forecasts = as.data.frame(t(forecasts))
    string_forecasts = format_csv(forecasts, col_names = FALSE)
    string_forecasts = strsplit(string_forecasts, "\n")[[1]]
    
    original_forecasts[i] = string_forecasts

    # write the arima forecasts to file
    #write.table(t(forecasts), file=output_file_name, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}
