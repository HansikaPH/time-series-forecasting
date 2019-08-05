library(forecast)
# read the data
file = "./datasets/text_data/CIF_2016/cif-dataset-o6.txt"
cif_dataset_o6 <- readLines(file)
cif_dataset_o6 <- strsplit(cif_dataset_o6, ',')

file = "./datasets/text_data/CIF_2016/cif-dataset-o12.txt"
cif_dataset_o12 <- readLines(file)
cif_dataset_o12 <- strsplit(cif_dataset_o12, ',')

output_file_name_o6 = "./results/ets_forecasts/cif2016_O6.txt"
output_file_name_o12 = "./results/ets_forecasts/cif2016_O12.txt"

unlink(output_file_name_o6)
unlink(output_file_name_o12)

# calculate the ets forecast
for (i in 1 : length(cif_dataset_o6)) {
    time_series = unlist(cif_dataset_o6[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

    fit = ets(ts(time_series, frequency=12))
    forecasts = forecast(fit, h=6)$mean

    forecasts[forecasts <0] = 0

    # write the ets forecasts to file
    write.table(t(forecasts), file=output_file_name_o6, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}

# calculate the ets forecast
for (i in 1 : length(cif_dataset_o12)) {
    time_series = unlist(cif_dataset_o12[i], use.names = FALSE)
    time_series = as.numeric(time_series[1 : length(time_series)])

    fit = ets(ts(time_series, frequency=12))
    forecasts = forecast(fit, h=12)$mean

    forecasts[forecasts <0] = 0

    # write the ets forecasts to file
    write.table(t(forecasts), file=output_file_name_o12, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}
