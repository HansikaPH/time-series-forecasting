library(forecast)
# read the data
kaggle_dataset <- read.csv(file = "./datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt", sep = ',', header = FALSE)
kaggle_dataset <- as.matrix(kaggle_dataset)

output_file_name = "./results/ets_forecasts/kaggle_web_traffic_non_seasonal.txt"

# clear the file content
unlink(output_file_name)

# calculate the ets forecasts
for (i in 1 : nrow(kaggle_dataset)) {
    forecasts = forecast(ets(ts(kaggle_dataset[i,], frequency=7)), h = 59)$mean

    forecasts[forecasts<0] <- 0
    forecasts = round(forecasts)

    forecasts[forecasts < 0] = 0

    # write the ets forecasts to file
    write.table(t(forecasts), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}