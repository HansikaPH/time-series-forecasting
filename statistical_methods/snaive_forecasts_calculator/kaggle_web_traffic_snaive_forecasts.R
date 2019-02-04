library(forecast)
# read the data
kaggle_dataset <- read.csv(file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt", sep = ',', header = FALSE)
kaggle_dataset <- as.matrix(kaggle_dataset)

output_file_name = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/snaive_forecasts/kaggle_web_traffic.txt"

# clear the file content
close(file(output_file_name, open = "w"))

# calculate the seasonal naive forecast
for (i in 1 : nrow(kaggle_dataset)) {
    fit <- snaive(ts(kaggle_dataset[i,], freq = 7), h = 59)

    # write the snaive forecasts to file
    write.table(t(fit$mean), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}