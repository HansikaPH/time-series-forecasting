library(forecast)
# read the data
file = "./datasets/text_data/Tourism/tourism_data.csv"
tourism_dataset <- readLines(file)
tourism_dataset <- strsplit(tourism_dataset, ',')

output_file_name = "./results/ets_forecasts/tourism.txt"

unlink(output_file_name)

# calculate the ets forecasts
for (i in 1 : length(tourism_dataset)) {
    time_series = unlist(tourism_dataset[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

    forecasts = forecast(ets(ts(time_series, frequency=12)), h = 24)$mean

    forecasts[forecasts < 0] = 0

    # write the ets forecasts to file
    write.table(t(forecasts), file=output_file_name, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}