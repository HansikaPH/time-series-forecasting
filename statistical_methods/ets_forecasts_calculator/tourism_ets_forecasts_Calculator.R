library(forecast)
# read the data
file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Tourism/tourism_data.csv"
tourism_dataset <- readLines(file)
tourism_dataset <- strsplit(tourism_dataset, ',')

output_file_name = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/ets_forecasts/tourism.txt"

# clear the file content
close(file(output_file_name, open = "w"))

# calculate the ets forecasts
for (i in 1 : length(tourism_dataset)) {
    time_series = unlist(tourism_dataset[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

    forecasts = forecast(ets(ts(time_series, frequency=12)), h = 24)$mean

    forecasts[forecasts < 0] = 0

    # write the ets forecasts to file
    write.table(t(forecasts), file=output_file_name, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}