library(forecast)
# read the data
file = "./datasets/text_data/Electricity/original_electricity_data_validation.csv"
electricity_dataset <- readLines(file)
electricity_dataset <- strsplit(electricity_dataset, ',')

output_file_name = "./results/ets_forecasts/electricity_validation.txt"

# clear the file content
close(file(output_file_name, open = "w"))

# calculate the ets forecasts
for (i in 1 : length(electricity_dataset)) {
    time_series = unlist(electricity_dataset[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

    forecasts = forecast(ets(ts(time_series, frequency=12)), h = 12)$mean

    forecasts[forecasts < 0] = 0

    # write the ets forecasts to file
    write.table(t(forecasts), file=output_file_name, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}