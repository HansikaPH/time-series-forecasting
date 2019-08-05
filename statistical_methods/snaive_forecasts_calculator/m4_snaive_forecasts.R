library(forecast)

# read the data
file = "./datasets/text_data/M4/Monthly-train.csv"
m4_dataset <- readLines(file)
m4_dataset <- strsplit(m4_dataset, ',')

output_file_path = "./results/snaive_forecasts/"
combined_snaive_forecasts_file = "./results/snaive_forecasts/m4.txt"

unlink(combined_snaive_forecasts_file)

# calculate the seasonal naive forecast
for (i in 2 : length(m4_dataset)) {
    time_series = unlist(m4_dataset[i], use.names = FALSE)
    time_series = as.numeric(time_series[2 : length(time_series)])
    fit <- snaive(ts(time_series, freq = 12), h = 18)

    # if (i - 1 <= 10016 && i - 1 >= 1) { #Macro Series
    #     output_file_name = paste(output_file_path, "m4_macro.txt")
    # }
    # else if (i - 1 <= 20991 && i - 1 > 10016) {
    #     output_file_name = paste(output_file_path, "m4_micro.txt")
    # }
    # else if (i - 1 <= 26719 && i - 1 > 20991) {
    #     output_file_name = paste(output_file_path, "m4_demo.txt")
    # }
    # else if (i - 1 <= 36736 && i - 1 > 26719) {
    #     output_file_name = paste(output_file_path, "m4_industry.txt")
    # }
    # else if (i - 1 <= 47723 && i - 1 > 36736) {
    #     output_file_name = paste(output_file_path, "m4_finance.txt")
    # }
    # else if (i - 1 > 47723) {
    #     output_file_name = paste(output_file_path, "m4_other.txt")
    # }
    #
    # # write the snaive forecasts to file
    # write.table(t(fit$mean), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
    write.table(t(fit$mean), file = combined_snaive_forecasts_file, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}

