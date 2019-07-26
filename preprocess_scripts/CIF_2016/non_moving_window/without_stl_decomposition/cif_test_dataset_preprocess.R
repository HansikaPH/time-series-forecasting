output_dir = "./datasets/text_data/CIF_2016/non_moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/CIF_2016/cif-dataset.txt"

cif_df = read.csv(file = input_file, sep = ';', header = FALSE)

names(cif_df)[4 : ncol(cif_df)] = paste('x', (1 : (ncol(cif_df) - 3)), sep = '_')
names(cif_df)[1] = "Series"
names(cif_df)[2] = "maxPredHorizon"

output_path6 = paste(output_dir, "cif_test_6.txt", sep = '/')
output_path12 = paste(output_dir, "cif_test_12.txt", sep = '/')

unlink(output_path6)
unlink(output_path12)

for (idr in 1 : nrow(cif_df)) {
    time_series = cif_df[idr,]
    series = as.character(time_series$Series)

    max_forecast_horizon = time_series$maxPredHorizon

    time_series = as.numeric(time_series[4 : (ncol(time_series))])
    time_series = time_series[! is.na(time_series)]
    mean = mean(time_series)
    time_series = time_series/mean
    time_series_log = log(time_series)
    time_series_length = length(time_series_log)

    sav_df = data.frame(id = paste(idr, '|i', sep = '')); #sav_df is the set of input values in the current window
    level = mean
    normalized_values = time_series_log
    sav_df = cbind(sav_df, t(normalized_values[1 : time_series_length])) #inputs: past values normalized by the level
    sav_df[, 'nyb'] = '|#' 
    sav_df[, 'level'] = level

    if (max_forecast_horizon == 6) {
        write.table(sav_df, file = output_path6, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    } else {
        write.table(sav_df, file = output_path12, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    }
}#through all series from one file
