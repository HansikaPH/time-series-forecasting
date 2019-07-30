library(forecast)

# read the data
file="./datasets/text_data/CIF_2016/cif-dataset.txt"
cif_dataset <- readLines(file)
cif_dataset <- strsplit(cif_dataset, ';')

output_file_name_o12 = "./results/snaive_forecasts/cif2016_O12.txt"
output_file_name_o6 = "./results/snaive_forecasts/cif2016_O6.txt"

# clear the file content
unlink(output_file_name_o12)
unlink(output_file_name_o6)

# calculate the seasonal naive forecast
for(i in 1:length(cif_dataset)){

    time_series = cif_dataset[i]
    series = as.numeric(unlist(time_series)[2])
    time_series = time_series[3:length(time_series)]

    if(series == 12){
        fit <- snaive(ts(as.numeric(unlist(time_series)), freq=12), h=12)

        # write the snaive forecasts to file
        write.table(t(fit$mean), file=output_file_name_o12, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)

    }else{
        fit <- snaive(ts(as.numeric(unlist(time_series)), freq=12), h=6)

        # write the snaive forecasts to file
        write.table(t(fit$mean), file=output_file_name_o6, row.names = F, col.names=F, sep=",", quote=F, append = TRUE)
    }

}