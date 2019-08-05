library(forecast)
# read the data
nn5_dataset <- read.csv(file = "./datasets/text_data/NN5/nn5_dataset.txt", sep = ',', header = FALSE)
nn5_dataset <- as.matrix(nn5_dataset)

output_file_name = "./results/snaive_forecasts/nn5.txt"

unlink(output_file_name)

# calculate the seasonal naive forecast
for (i in 1 : nrow(nn5_dataset)) {
    fit <- snaive(ts(nn5_dataset[i,], freq = 7), h = 56)

    # write the snaive forecasts to file
    write.table(t(fit$mean), file = output_file_name, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
}