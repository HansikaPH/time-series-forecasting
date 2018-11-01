library(forecast)

# read the data
file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M3/Train_Dataset.csv"
m3_dataset <- readLines(file)
m3_dataset <- strsplit(m3_dataset, ',')

output_file_path = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/snaive_forecasts/"

# clear the file content
# close( file( output_file_name, open="w" ) )

# calculate the seasonal naive forecast
for (i in 1 : length(m3_dataset)) {
    time_series = unlist(m3_dataset[i], use.names = FALSE)
    time_series = as.numeric(time_series[2 : length(time_series)])
    fit <- snaive(ts(time_series, freq = 12), h = 18)

    if (i <= 474 && i >= 1) { #Macro Series
        output_file_name = paste(output_file_path, "m3_micro.txt")
    }
    else if (i <= 808 && i > 474) {
        output_file_name = paste(output_file_path, "m3_industry.txt")
    }
    else if (i <= 1120 && i > 808) {
        output_file_name = paste(output_file_path, "m3_macro.txt")
    }
    else if (i <= 1265 && i > 1120) {
        output_file_name = paste(output_file_path, "m3_finance.txt")
    }
    else if (i <= 1376 && i > 1265) {
        output_file_name = paste(output_file_path, "m3_demo.txt")
    }
    else if (i > 1376) {
        output_file_name = paste(output_file_path, "m3_other.txt")
    }

    # write the snaive forecasts to file
    write.table(t(fit$mean), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}

