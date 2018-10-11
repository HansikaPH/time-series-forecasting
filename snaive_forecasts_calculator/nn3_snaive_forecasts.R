library(forecast)

# read the data
file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN3/NN3_DATASET.csv"
nn3_dataset <- readLines(file)
nn3_dataset <- strsplit(nn3_dataset, ',')

output_file_name = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/snaive_forecasts/nn3.txt"

# clear the file content
close( file( output_file_name, open="w" ) )

# calculate the seasonal naive forecast
for(i in 1:length(nn3_dataset)){
    fit <- snaive(ts(as.numeric(unlist(nn3_dataset[i])), freq=12), h=18)

    # write the snaive forecasts to file
    write.table(t(fit$mean), file=output_file_name, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
}