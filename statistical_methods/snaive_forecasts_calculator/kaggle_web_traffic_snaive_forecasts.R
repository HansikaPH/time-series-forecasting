library(forecast)
# read the data
kaggle_dataset <- read.csv(file = "./datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt", sep = ',', header = FALSE)
kaggle_dataset <- as.matrix(kaggle_dataset)

output_file_name = "./results/snaive_forecasts/kaggle_web_traffic.txt"

unlink(output_file_name)

# calculate the seasonal naive forecast
for (i in 1 : nrow(kaggle_dataset)) {
    fit <- snaive(ts(kaggle_dataset[i,], freq = 7), h = 59)

    # write the snaive forecasts to file
    write.table(t(fit$mean), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}